import os
import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchResults
from litellm import completion
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL = os.getenv("MODEL", "ollama_chat/phi3")
API_BASE = os.getenv("API_BASE", "http://localhost:11434")
MAX_STEPS = int(os.getenv("MAX_STEPS", 20))
CHUNK_SIZE = 5
MIN_UNIQUE_RESULTS = 10

# DuckDuckGo search tool with list output format for consistent results
search_tool = DuckDuckGoSearchResults(output_format="list")

# Helper function to extract JSON from LLM response
def extract_json(response: str) -> dict:
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    return json.loads(match.group()) if match else {}

# LLM call function
def llm_call(prompt: str) -> str:
    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_base=API_BASE
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "Error: Unable to process the request."

# Pydantic models
class SceneInfo(BaseModel):
    scene: str
    studio: str
    link: str

class State(BaseModel):
    query: str = ""
    refined_query: str = ""
    databases: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, str]] = Field(default_factory=list)
    report: List[Dict[str, str]] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = MAX_STEPS
    next_node: str = ""
    consecutive_no_new_results: int = 0

# Node Functions
def initialize_state_node(state: State) -> Dict[str, Any]:
    """Initialize the state with the original query."""
    return {
        "refined_query": state.query,
        "step_count": 0,
        "next_node": "query_refinement",
        "consecutive_no_new_results": 0
    }

def query_refinement_node(state: State) -> Dict[str, Any]:
    """Refine the search query using the LLM."""
    prompt = (
        "You are a data extraction assistant. Given the query 'scenes involving Rebecca Linares in fetish-related media content', "
        "suggest a refined search query to find specific scene information on databases or websites. "
        "Focus on keywords likely to appear in database entries or listings. "
        "Output JSON with key 'refined_query'."
    )
    response = llm_call(prompt)
    data = extract_json(response)
    refined_query = data.get("refined_query", state.refined_query)
    return {"refined_query": refined_query}

def database_selection_node(state: State) -> Dict[str, Any]:
    """Generate a search query to find relevant databases or websites."""
    prompt = (
        "Generate a search query to find databases or websites containing information about media content featuring performers like Rebecca Linares. "
        "Use terms like 'database', 'archive', or known platforms. Output JSON with key 'search_query'."
    )
    response = llm_call(prompt)
    data = extract_json(response)
    search_query = data.get("search_query", "adult film database")
    # Perform search to find databases
    try:
        results = search_tool.run(search_query)
        domains = [urlparse(r['link']).netloc for r in results if r.get('link')]
    except Exception as e:
        print(f"Error searching for databases: {e}")
        domains = []
    return {"databases": domains}

def search_node(state: State) -> Dict[str, Any]:
    """Perform searches on selected databases or general search."""
    combined_results = []
    for db in state.databases:
        query = f"site:{db} {state.refined_query}"
        try:
            results = search_tool.run(query)
            combined_results.extend(results)
        except Exception as e:
            print(f"Error searching site:{db}: {e}")
    # General search fallback
    try:
        general_results = search_tool.run(state.refined_query)
        combined_results.extend(general_results)
    except Exception as e:
        print(f"Error in general search: {e}")
    return {"search_results": combined_results}

def data_processing_node(state: State) -> Dict[str, Any]:
    """Extract scene information from search results using LLM."""
    search_results = state.search_results
    new_scenes = []
    for i in range(0, len(search_results), CHUNK_SIZE):
        chunk = search_results[i:i+CHUNK_SIZE]
        results_text = "\n".join([f"{idx+1}. Title: {r['title']}\n   Link: {r['link']}\n   Snippet: {r['snippet']}" for idx, r in enumerate(chunk)])
        prompt = (
            "You are a data extraction assistant. From the following search results, extract information about specific scenes involving Rebecca Linares. "
            "For each relevant result, provide the scene name, studio, and link if available. "
            "Return a JSON array of objects with keys 'scene', 'studio', 'link'. If a result is not relevant, omit it."
            f"\n\nResults:\n{results_text}"
        )
        response = llm_call(prompt)
        try:
            extracted = json.loads(response)
            if not isinstance(extracted, list):
                extracted = []
        except json.JSONDecodeError:
            extracted = []
        for item in extracted:
            try:
                scene_info = SceneInfo(**item)
                new_scene = {
                    "scene": scene_info.scene,
                    "studio": scene_info.studio,
                    "link": scene_info.link
                }
                if new_scene not in state.report:
                    new_scenes.append(new_scene)
            except ValidationError:
                continue
    state.report.extend(new_scenes)
    consecutive_no_new = 0 if new_scenes else state.consecutive_no_new_results + 1
    return {"report": state.report, "consecutive_no_new_results": consecutive_no_new}

def quality_check_node(state: State) -> Dict[str, Any]:
    """Decide if enough results are collected or if refinement is needed."""
    state.step_count += 1
    if len(state.report) >= MIN_UNIQUE_RESULTS or state.step_count >= state.max_steps or state.consecutive_no_new_results >= 3:
        next_node = "response"
    else:
        next_node = "query_refinement"
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    """Generate the final report."""
    report = {
        "Original Query": state.query,
        "Results": state.report,
        "Conclusion": f"Collected {len(state.report)} unique scenes."
    }
    with open("research_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Final report generated.")
    return {"report": report}

# Build the graph
workflow = StateGraph(State)
workflow.add_node("initialize", initialize_state_node)
workflow.add_node("query_refinement", query_refinement_node)
workflow.add_node("database_selection", database_selection_node)
workflow.add_node("search", search_node)
workflow.add_node("data_processing", data_processing_node)
workflow.add_node("quality_check", quality_check_node)
workflow.add_node("response", response_node)

# Define edges
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

# Compile the graph
graph = workflow.compile()

# Main execution
def main():
    initial_state = State(query="scenes involving Rebecca Linares in fetish-related adult films")
    final_state = graph.invoke(initial_state)
    print(json.dumps(final_state["report"], indent=2))

if __name__ == "__main__":
    main()