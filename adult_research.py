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
MAX_STEPS = int(os.getenv("MAX_STEPS", 20))  # Adjust as needed
CHUNK_SIZE = 5  # Number of search results to process per LLM call

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

    # (Optional) Allows extra fields if the graph returns keys
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

def extract_json(response: str):
    """
    Extract JSON from the response string.
    This function removes Markdown-style triple backticks if present,
    and then attempts to parse the result.
    Works for both JSON objects and JSON arrays.
    """
    response = response.strip()
    # Remove markdown formatting if present
    markdown_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    match = markdown_pattern.search(response)
    if match:
        json_str = match.group(1)
    else:
        json_str = response

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

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
    query = state.original_query
    print(f"Initializing state with query: {query}")
    return {
        "report": {"query": query, "results": [], "sources": [], "notes": ""},
        "current_query": query,
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
    Asks the LLM for a refined web search query. If it's identical to previous queries,
    we can detect that to avoid infinite loops.
    """
    current_query = state.current_query
    result_count = len(state.report.results)

    prompt = (
        f"You are assisting a researcher investigating adult content featuring Rebecca Linares. "
        f"Given the current search query: '{current_query}', and the fact that {result_count} "
        f"relevant results have been found so far, please suggest a refined search query. "
        f"Try to include specific keywords such as 'scene', 'studio', or other relevant terms "
        f"that might yield more specific results. Output JSON with key 'refined_query'."
    )
    response = llm_call(prompt)
    extracted_json = extract_json(response)
    try:
        refined_query_data = RefinedQuery(**(extracted_json or {}))
        refined_query = refined_query_data.refined_query.strip()
    except ValidationError as e:
        print(f"Validation error in query refinement: {e}")
        refined_query = current_query
    
    print(f"Refined query: {refined_query}")

    new_previous_queries = state.previous_queries + [refined_query]
    return {
        "current_query": refined_query,
        "previous_queries": new_previous_queries
    }

def database_selection_node(state: State) -> Dict[str, Any]:
    """
    Asks the LLM to generate a search query for adult websites or databases
    that might contain info about the current query, then runs a search
    to identify relevant domains.
    """
    current_query = state.current_query
    prompt = (
        f"Generate a search query that might reveal websites or databases with "
        f"information about '{current_query}'. Use terms like 'adult database', "
        f"'porn site', or specific known adult content platforms. Output JSON "
        f"with key 'search_query'."
    )
    response = llm_call(prompt)
    extracted_json = extract_json(response)

    try:
        search_query_data = SearchQuery(**(extracted_json or {}))
        search_query = search_query_data.search_query.strip()
    except ValidationError as e:
        print(f"Validation error in database selection: {e}")
        search_query = current_query

    print(f"Search query for databases: {search_query}")

    # Run the search using DuckDuckGoSearchResults
    try:
        results = search_tool.invoke(search_query)
        print(f"Search results for databases: {results}")
    except Exception as e:
        print(f"Error during search_tool.invoke: {e}")
        results = []

    # Identify domains from the results
    domains = []
    for result in results:
        url = result.get('link', '')
        domain = urlparse(url).netloc
        if domain and domain not in domains:
            domains.append(domain)

    # Combine with existing useful_databases, remove duplicates
    all_domains = list(set(domains + state.useful_databases))
    # Keep a "general" fallback
    databases = [{"type": "site", "url": domain} for domain in all_domains] + [{"type": "general"}]
    print(f"Selected databases: {databases}")

    return {"databases": databases}

def search_node(state: State) -> Dict[str, Any]:
    """
    Searches each selected database (domain) or performs a general search for the current_query.
    """
    combined_results = []
    for db in state.databases:
        if db["type"] == "site":
            query = f"site:{db['url']} {state.current_query}"
        else:
            query = state.current_query
        
        print(f"Searching with query: {query}")
        try:
            results = search_tool.invoke(query)
            print(f"Search results: {results}")
            combined_results.extend(results)
        except Exception as e:
            print(f"Error during search_tool.invoke for query '{query}': {e}")

    return {"search_results": combined_results}

def data_processing_node(state: State) -> Dict[str, Any]:
    """
    Takes the raw search results, calls the LLM to extract scene info,
    and appends them to the report if relevant.
    """
    report = state.report
    useful_databases = state.useful_databases
    search_results = state.search_results

    print(f"Processing {len(search_results)} search results")
    new_results_found = False

    for i in range(0, len(search_results), CHUNK_SIZE):
        chunk = search_results[i:i + CHUNK_SIZE]
        # Format the prompt for the chunk
        results_text = "\n".join([
            f"{idx+1}. Title: {res.get('title','')}\n   Link: {res.get('link','')}\n   Snippet: {res.get('snippet','')}"
            for idx, res in enumerate(chunk)
        ])

        prompt = (
            "You are extracting data about specific adult film scenes featuring Rebecca Linares.\n\n"
            "Below are some search results:\n"
            f"{results_text}\n\n"
            "For each result that specifically mentions or suggests an adult scene with Rebecca Linares, "
            "extract:\n- scene name (as 'scene')\n- studio or production company (as 'studio')\n- direct link (as 'link')\n\n"
            "Return a JSON array of objects with keys [scene, studio, link]. "
            "If a result does not appear to reference an actual scene with Rebecca Linares, omit it."
        )
        response = llm_call(prompt)
        extracted_scenes = extract_json(response)
        if not isinstance(extracted_scenes, list):
            extracted_scenes = []

        for scene_data in extracted_scenes:
            try:
                scene_info = SceneInfo(**scene_data)
                scene = {
                    "scene": scene_info.scene,
                    "studio": scene_info.studio,
                    "link": scene_info.link
                }
                domain = urlparse(scene["link"]).netloc
                scene["verification"] = f"Source: {domain}, manual verification needed"

                # Only add if not already in results
                if not any(r.get("link") == scene["link"] for r in report.results):
                    report.results.append(scene)
                    new_results_found = True
                    print(f"Added scene: {scene['scene']} (Studio: {scene['studio']}, Link: {scene['link']})")

                    if domain and domain not in report.sources:
                        report.sources.append(domain)
                        print(f"Added source: {domain}")

                    if domain and domain not in useful_databases:
                        useful_databases.append(domain)

            except ValidationError as e:
                print(f"Validation error for scene data {scene_data}: {e}")
                continue

    if new_results_found:
        print("New relevant results found. Resetting consecutive_no_new_results.")
        consecutive_count = 0
    else:
        consecutive_count = state.consecutive_no_new_results + 1
        print(f"No new results this round. consecutive_no_new_results = {consecutive_count}")

    print(f"Processed results. Current result count: {len(report.results)}")
    return {
        "report": report,
        "search_results": [],
        "useful_databases": useful_databases,
        "consecutive_no_new_results": consecutive_count
    }

def quality_check_node(state: State) -> Dict[str, Any]:
    """
    Checks if we have enough results or if we've tried too many times.
    If we haven't found anything new for N times, we also stop early.
    """
    state.step_count += 1
    target_results = 3  # For demonstration, stop after we find 3 relevant results
    current_results = len(state.report.results)
    no_result_limit = 3  # Stop if 3 consecutive tries had no new results

    if (
        state.step_count >= state.max_steps
        or current_results >= target_results
        or state.consecutive_no_new_results >= no_result_limit
    ):
        next_node = "response"
        if state.consecutive_no_new_results >= no_result_limit and current_results == 0:
            state.report.notes = "No relevant results found after multiple refinements."
    else:
        next_node = "query_refinement"

    print(f"Step {state.step_count}/{state.max_steps}: {current_results} results found. Next node: {next_node}")
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    """
    Final node: outputs the final aggregated report and stops.
    """
    print("Generating final report")
    report = state.report
    print(f"Final report - Query: {report.query}, Results: {len(report.results)}, Sources: {len(report.sources)}")
    return {"report": report}

###############################################################################
# Graph Definition
###############################################################################

from langgraph.pregel import GraphRecursionError

search_tool = DuckDuckGoSearchResults(output_format="list")

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
    lambda state: state.next_node,
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
    final_state = State(**dict(final_data))
    print("Agent execution completed")
    return final_state.report

if __name__ == "__main__":
    sample_query = "Find fetish scenes with Rebecca Linares"
    report = run_agent(sample_query)
    print("\n=== Final Report ===")
    print(f"Query: {report.query}")
    print("Results:")
    for result in report.results:
        print(f"- {result['scene']} (Studio: {result['studio']}, Link: {result['link']}, Verification: {result['verification']})")
    print(f"Sources: {report.sources}")
    print(f"Notes: {report.notes}")
