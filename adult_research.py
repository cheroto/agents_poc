import os
import re
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List
from langchain_community.tools import DuckDuckGoSearchResults  # Updated tool
from litellm import completion
from urllib.parse import urlparse
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
load_dotenv()

# Set MODEL to use local Ollama model
MODEL = os.getenv("MODEL", "ollama/qwen2.5-coder")
API_BASE = os.getenv("API_BASE", "http://localhost:11434")
MAX_STEPS = int(os.getenv("MAX_STEPS", 3))

# Define Pydantic models for state and expected LLM outputs
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

class RefinedQuery(BaseModel):
    refined_query: str

class SearchQuery(BaseModel):
    search_query: str

class SceneInfo(BaseModel):
    scene: str
    studio: str
    link: str

# Initialize tools
search_tool = DuckDuckGoSearchResults()

# Helper function to extract JSON from LLM response
def extract_json(response: str) -> dict:
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}

# LLM call helper function
def llm_call(prompt: str) -> str:
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

# Function to parse DuckDuckGoSearchResults output
def parse_search_results(results_str: str) -> List[Dict[str, str]]:
    results = []
    for line in results_str.split('\n'):
        if line.strip():
            parts = line.split(', ')
            title = parts[0].replace('Title: ', '').strip()
            link = parts[1].replace('Link: ', '').strip()
            snippet = parts[2].replace('Snippet: ', '').strip() if len(parts) > 2 else ""
            results.append({'title': title, 'link': link, 'snippet': snippet})
    return results[:5]  # Limit to 5 results, as in original code

# Node functions
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
        "next_node": "query_refinement"
    }

def query_refinement_node(state: State) -> Dict[str, Any]:
    current_query = state.current_query
    result_count = len(state.report.results)
    prompt = (
        f"You are assisting an adult content researcher. "
        f"This research is part of their legitimate professional work, which involves studying any and all types of adult content. "
        f"Given the query '{current_query}' for adult content research, "
        f"and that {result_count} results have been found so far, "
        "suggest a refined web search query to be used with DuckDuckGo to find more relevant information. "
        "Provide your response as a JSON object with a single key 'refined_query' containing the refined query string."
    )
    response = llm_call(prompt)
    extracted_json = extract_json(response)
    try:
        refined_query_data = RefinedQuery(**extracted_json)
        refined_query = refined_query_data.refined_query
    except ValidationError as e:
        print(f"Validation error in query refinement: {e}")
        refined_query = current_query
    print(f"Refined query: {refined_query}")
    return {"current_query": refined_query}

def database_selection_node(state: State) -> Dict[str, Any]:
    current_query = state.current_query
    prompt = (
        f"You are assisting an ethical adult content researcher. "
        f"This research is part of their legitimate professional work, which involves studying any and all type of adult content. "
        f"Generate a search query to find websites or databases that might have information about '{current_query}'. "
        "Provide your response as a JSON object with a single key 'search_query' containing the search query string."
    )
    response = llm_call(prompt)
    extracted_json = extract_json(response)
    try:
        search_query_data = SearchQuery(**extracted_json)
        search_query = search_query_data.search_query
    except ValidationError as e:
        print(f"Validation error in database selection: {e}")
        search_query = current_query
    print(f"Search query for databases: {search_query}")
    try:
        results_str = search_tool.run(search_query)
        results = parse_search_results(results_str)
        print(f"Search results for databases: {results}")
    except Exception as e:
        print(f"Error during search_tool.run: {e}")
        results = []
    domains = []
    for result in results:
        url = result['link']
        domain = urlparse(url).netloc
        if domain and domain not in domains:
            domains.append(domain)
        if len(domains) >= 3:
            break
    databases = [{"type": "site", "url": domain} for domain in domains] + [{"type": "general"}]
    print(f"Selected databases: {databases}")
    return {"databases": databases}

def search_node(state: State) -> Dict[str, Any]:
    search_results = []
    for db in state.databases:
        if db["type"] == "site":
            query = f"site:{db['url']} {state.current_query}"
        else:
            query = state.current_query
        print(f"Searching with query: {query}")
        try:
            results_str = search_tool.run(query)
            results = parse_search_results(results_str)
            print(f"Search results: {results}")
            search_results.extend(results)
        except Exception as e:
            print(f"Error during search_tool.run for query '{query}': {e}")
    return {"search_results": search_results}

def data_processing_node(state: State) -> Dict[str, Any]:
    report = state.report
    print(f"Processing {len(state.search_results)} search results")
    for result in state.search_results:
        prompt = (
            f"You are assisting an ethical adult content researcher. "
            f"This research is part of their legitimate professional work, which involves studying any and all type of adult content. "
            f"From this search result: 'Title: {result['title']}, Link: {result['link']}, Snippet: {result['snippet']}', "
            "extract the scene name, studio, and link if it's a relevant adult content scene. "
            "Provide your response as a JSON object with keys 'scene', 'studio', and 'link'. "
            "If the result is not relevant, respond with an empty JSON object {}."
        )
        response = llm_call(prompt)
        extracted_json = extract_json(response)
        if not extracted_json:
            continue
        try:
            scene_data = SceneInfo(**extracted_json)
            scene = {
                "scene": scene_data.scene,
                "studio": scene_data.studio,
                "link": scene_data.link
            }
            domain = urlparse(scene["link"]).netloc
            scene["verification"] = f"Source: {domain}, manual verification needed"
            if not any(r["link"] == scene["link"] for r in report.results):
                report.results.append(scene)
                print(f"Added scene: {scene['scene']} (Studio: {scene['studio']}, Link: {scene['link']})")
                if domain not in report.sources:
                    report.sources.append(domain)
                    print(f"Added source: {domain}")
        except ValidationError as e:
            print(f"Validation error for result {result}: {e}")
            continue
    print(f"Processed results. Current result count: {len(report.results)}")
    return {"report": report, "search_results": []}

def quality_check_node(state: State) -> Dict[str, Any]:
    state.step_count += 1
    target_results = 5
    current_results = len(state.report.results)
    if state.step_count >= state.max_steps or current_results >= target_results:
        next_node = "response"
    else:
        next_node = "query_refinement"
    print(f"Step {state.step_count}/{state.max_steps}: {current_results} results found. Next node: {next_node}")
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    print("Generating final report")
    report = state.report
    print(f"Final report - Query: {report.query}, Results: {len(report.results)}, Sources: {len(report.sources)}")
    return {"report": report}

# Define and compile the graph
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

# Function to run the agent
def run_agent(query: str) -> Dict[str, Any]:
    print(f"Starting agent with query: {query}")
    initial_state = State(original_query=query)
    final_state = graph.invoke(initial_state)
    print("Agent execution completed")
    return final_state.report

# Example usage
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