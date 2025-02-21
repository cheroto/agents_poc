import os
import re
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List
from langchain_community.tools import DuckDuckGoSearchResults
from litellm import completion
from urllib.parse import urlparse
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
load_dotenv()

# Set MODEL to use local Ollama model (or configure as needed)
MODEL = os.getenv("MODEL", "ollama/qwen2.5-coder")
API_BASE = os.getenv("API_BASE", "http://localhost:11434")

# Adjust as needed
MAX_STEPS = int(os.getenv("MAX_STEPS", 20))
CHUNK_SIZE = 5

###############################################################################
# Pydantic models
###############################################################################

class Report(BaseModel):
    query: str = ""
    results: List[Dict[str, str]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    notes: str = ""

class State(BaseModel):
    report: Report = Field(default_factory=Report)
    current_query: str = ""
    databases: List[Dict[str, str]] = Field(default_factory=list)
    search_results: List[Dict[str, str]] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = MAX_STEPS
    next_node: str = ""
    original_query: str = ""
    useful_databases: List[str] = Field(default_factory=list)
    consecutive_no_new_results: int = 0
    previous_queries: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"

class RefinedQuery(BaseModel):
    refined_query: str

class SearchQuery(BaseModel):
    search_query: str

class SceneInfo(BaseModel):
    scene: str
    studio: str
    link: str

###############################################################################
# Helper functions
###############################################################################

# By default, do not specify backend="api", which often returns fewer results.
search_tool = DuckDuckGoSearchResults(output_format="list")

def extract_json(response: str) -> dict:
    """
    Extract the FIRST {...} JSON object found in the response string.
    Handles triple backticks or other wrappers by a simple regex match.
    """
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}

def llm_call(prompt: str) -> str:
    """
    Calls the LLM with the provided prompt, returning the text of the LLM response.
    """
    print(f"Calling LLM with prompt: {prompt}")
    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_base=API_BASE
        )
        result = response.choices[0].message.content
        print(f"LLM response: {result}")
        return result
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "Error: Unable to process the request at this time."

###############################################################################
# Node Functions
###############################################################################

def initialize_state_node(state: State) -> Dict[str, Any]:
    """
    Node 1: Initialize the state. We do NOT return 'report' here; we only do in-place updates.
    """
    print(f"Initializing state with query: {state.original_query}")
    state.report.query = state.original_query
    return {
        "current_query": state.original_query,
        "databases": [],
        "search_results": [],
        "step_count": 0,
        "max_steps": MAX_STEPS,
        "next_node": "query_refinement",
        "useful_databases": [],
        "consecutive_no_new_results": 0,
        "previous_queries": []
    }

def query_refinement_node(state: State) -> Dict[str, Any]:
    """
    Node 2: Ask LLM for a refined query. We do in-place updates on state as needed.
    """
    current_query = state.current_query
    result_count = len(state.report.results)

    prompt = (
        f"You are assisting a researcher investigating adult content featuring Rebecca Linares. "
        f"Given the current search query: '{current_query}', and {result_count} relevant results, "
        "suggest a refined web search query with specific keywords (e.g. 'scene', 'studio', 'fetish') "
        "Output JSON with key 'refined_query'."
    )
    response = llm_call(prompt)
    data = extract_json(response)
    try:
        refined = RefinedQuery(**data)
        refined_query = refined.refined_query.strip()
    except ValidationError as e:
        print(f"Validation error in query refinement: {e}")
        refined_query = current_query

    print(f"Refined query: {refined_query}")
    state.previous_queries.append(refined_query)

    return {
        "current_query": refined_query,
        "previous_queries": state.previous_queries
    }

def database_selection_node(state: State) -> Dict[str, Any]:
    """
    Node 3: LLM generates a search query to find adult content websites or DBs. Then we do a short search.
    """
    current_query = state.current_query
    prompt = (
        f"Generate a search query that might reveal adult websites or databases for '{current_query}'. "
        "Use terms like 'adult database', 'porn site', or known adult content platforms. "
        "Output JSON with key 'search_query'."
    )
    response = llm_call(prompt)
    data = extract_json(response)

    try:
        sq = SearchQuery(**data)
        search_query = sq.search_query.strip()
    except ValidationError as e:
        print(f"Validation error in database selection: {e}")
        search_query = current_query

    print(f"Search query for databases: {search_query}")

    # Run the search
    try:
        results = search_tool.invoke(search_query)
        print(f"Search results for databases: {results}")
    except Exception as e:
        print(f"Error during search_tool.invoke: {e}")
        results = []

    # Identify unique domains
    domains = []
    for r in results:
        url = (r.get('link') or "").strip()
        domain = urlparse(url).netloc
        if domain and domain not in domains:
            domains.append(domain)

    # Merge with known DBs
    all_domains = list(set(domains + state.useful_databases))
    new_db = [{"type": "site", "url": d} for d in all_domains] + [{"type": "general"}]
    print(f"Selected databases: {new_db}")

    return {"databases": new_db}

def search_node(state: State) -> Dict[str, Any]:
    """
    Node 4: Search each domain or do a general search, collect combined results in one list.
    """
    combined = []
    for db in state.databases:
        if db["type"] == "site":
            query = f"site:{db['url']} {state.current_query}"
        else:
            query = state.current_query
        
        print(f"Searching with query: {query}")
        try:
            results = search_tool.invoke(query)
            print(f"Search results: {results}")
            combined.extend(results)
        except Exception as e:
            print(f"Error searching '{query}': {e}")

    return {"search_results": combined}

def data_processing_node(state: State) -> Dict[str, Any]:
    """
    Node 5: For each chunk of search_results, call LLM to parse out scene/studio/link,
    then store in state.report.results if new.
    """
    search_results = state.search_results
    print(f"Processing {len(search_results)} search results")
    new_found = False

    for i in range(0, len(search_results), CHUNK_SIZE):
        chunk = search_results[i:i+CHUNK_SIZE]
        results_text = "\n".join([
            f"{idx+1}. Title: {r.get('title','')}\n   Link: {r.get('link','')}\n   Snippet: {r.get('snippet','')}"
            for idx, r in enumerate(chunk)
        ])

        prompt = (
            "You are extracting data about specific adult film scenes featuring Rebecca Linares.\n\n"
            f"Below are search results:\n{results_text}\n\n"
            "For each result referencing a specific scene with Rebecca Linares, extract:\n"
            "- scene (as 'scene')\n"
            "- studio or production company (as 'studio')\n"
            "- direct link (as 'link')\n\n"
            "Return a JSON array of objects with keys [scene, studio, link]. "
            "If a result is not relevant, omit it."
        )
        response = llm_call(prompt)
        try:
            extracted_scenes = json.loads(response)
            if not isinstance(extracted_scenes, list):
                extracted_scenes = []
        except json.JSONDecodeError:
            extracted_scenes = []

        for scene_data in extracted_scenes:
            try:
                scene_info = SceneInfo(**scene_data)
                # Clean/truncate whitespace
                link_str = scene_info.link.strip()
                domain = urlparse(link_str).netloc.strip()

                # Build the new scene record
                new_scene = {
                    "scene": scene_info.scene.strip(),
                    "studio": scene_info.studio.strip(),
                    "link": link_str,
                    "verification": f"Source: {domain}, manual verification needed"
                }

                # Check duplicates
                already_exists = any(
                    (r.get("link") or "").strip() == link_str
                    for r in state.report.results
                )
                if already_exists:
                    print(f"Duplicate link found, skipping: {link_str}")
                else:
                    state.report.results.append(new_scene)
                    new_found = True
                    print(f"Added scene: {new_scene['scene']} (Link: {link_str})")

                    # Add domain to sources
                    if domain and domain not in state.report.sources:
                        state.report.sources.append(domain)
                        print(f"Added source: {domain}")

                    # Also add domain to useful_databases
                    if domain and domain not in state.useful_databases:
                        state.useful_databases.append(domain)

            except ValidationError as e:
                print(f"Scene data validation error: {e}")
                continue

    if new_found:
        print("New relevant results found. Resetting consecutive_no_new_results.")
        consecutive = 0
    else:
        consecutive = state.consecutive_no_new_results + 1
        print(f"No new results this round. consecutive_no_new_results = {consecutive}")

    print(f"Processed results. Current result count: {len(state.report.results)}")

    # DO NOT RETURN 'report' here! Just return updated fields.
    return {
        "search_results": [],
        "consecutive_no_new_results": consecutive,
        "useful_databases": state.useful_databases
    }

def quality_check_node(state: State) -> Dict[str, Any]:
    """
    Node 6: Decide if we have enough results or if we should continue refining.
    """
    state.step_count += 1
    target = 3
    current_count = len(state.report.results)
    no_result_limit = 3

    if (
        state.step_count >= state.max_steps
        or current_count >= target
        or state.consecutive_no_new_results >= no_result_limit
    ):
        next_node = "response"
        if state.consecutive_no_new_results >= no_result_limit and current_count == 0:
            state.report.notes = "No relevant results found after multiple refinements."
    else:
        next_node = "query_refinement"

    print(f"Step {state.step_count}/{state.max_steps}: {current_count} results found. Next node: {next_node}")
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    """
    Node 7 (Final): Output the final aggregated report as a dictionary.
    """
    print("Generating final report")
    print(f"Final report - Query: {state.report.query}, "
          f"Results: {len(state.report.results)}, "
          f"Sources: {len(state.report.sources)}")
    return {"report": state.report.dict()}

###############################################################################
# Graph Definition
###############################################################################

from langgraph.pregel import GraphRecursionError

workflow = StateGraph(State)
workflow.add_node("initialize", initialize_state_node)
workflow.add_node("query_refinement", query_refinement_node)
workflow.add_node("database_selection", database_selection_node)
workflow.add_node("search", search_node)
workflow.add_node("data_processing", data_processing_node)
workflow.add_node("quality_check", quality_check_node)
workflow.add_node("response", response_node)

workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "query_refinement")
workflow.add_edge("query_refinement", "database_selection")
workflow.add_edge("database_selection", "search")
workflow.add_edge("search", "data_processing")
workflow.add_edge("data_processing", "quality_check")

workflow.add_conditional_edges(
    "quality_check",
    lambda s: s.next_node,
    {"query_refinement": "query_refinement", "response": "response"}
)

workflow.add_edge("response", END)
graph = workflow.compile()

###############################################################################
# Main Agent Function
###############################################################################

def run_agent(query: str) -> Dict[str, Any]:
    print(f"Starting agent with query: {query}")
    initial_state = State(original_query=query)
    final_data = graph.invoke(initial_state)
    # final_data["report"] is the dictionary from response_node
    print("Agent execution completed")
    return final_data["report"]

if __name__ == "__main__":
    sample_query = "Find fetish scenes with Rebecca Linares"
    final_report = run_agent(sample_query)
    print("\n=== Final Report ===")
    print(f"Query: {final_report['query']}")
    print("Results:")
    for r in final_report["results"]:
        print(f"- Scene: {r['scene']} | Studio: {r['studio']} | Link: {r['link']} | Verification: {r['verification']}")
    print(f"Sources: {final_report['sources']}")
    print(f"Notes: {final_report['notes']}")
