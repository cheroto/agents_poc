import os
import re
import json
import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List
from langchain_community.tools import DuckDuckGoSearchResults
from litellm import completion
from urllib.parse import urlparse
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Configuration Section
# =============================================================================
MODEL = os.getenv("MODEL", "ollama/qwen2.5-coder")
API_BASE = os.getenv("API_BASE", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_STEPS = int(os.getenv("MAX_STEPS", 20))  # Maximum steps for the workflow
CHUNK_SIZE = 1  # Number of search results to process per LLM call

TARGET_RESULTS = 3       # Number of results required to stop search
NO_RESULT_LIMIT = 6      # Number of consecutive iterations with no new results before stopping
RECURSION_LIMIT = 100     # Recursion limit for the graph
USAGE_THRESHOLD = 3      # Threshold for domain usage to trigger exclusion

QUERY = "Find bisexual mmf scenes with Rebecca Linares"


# Initialize search tool
search_tool = DuckDuckGoSearchResults(output_format="list")

# =============================================================================
# Pydantic models
# =============================================================================
class Report(BaseModel):
    query: str = ""
    results: List[Dict[str, str]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    notes: str = ""

class State(BaseModel):
    report: Report = Field(default_factory=Report)
    current_query: str = ""
    search_results: List[Dict[str, str]] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = MAX_STEPS
    next_node: str = ""
    original_query: str = ""
    useful_databases: List[str] = Field(default_factory=list)
    consecutive_no_new_results: int = 0
    previous_queries: List[str] = Field(default_factory=list)
    # Track how many times we have used or seen each domain
    domain_usage: Dict[str, int] = Field(default_factory=dict)
    # A list of domains to exclude in subsequent searches (e.g., if we've over-used them)
    excluded_domains: List[str] = Field(default_factory=list)
    # Added search cache to store previous search results
    search_cache: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)

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

# =============================================================================
# Helper functions
# =============================================================================
def extract_json(response: str):
    """
    Extract JSON from the response string.
    This function removes Markdown-style triple backticks if present,
    and then attempts to parse the result.
    Works for both JSON objects and JSON arrays.
    """
    response = response.strip()
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
    A system message is added to instruct the model to follow instructions exactly.
    """
    system_message = "Follow the user's instructions precisely. Respond exactly as requested without any additional commentary."
    print(f"Calling LLM with prompt: {prompt}")
    try:
        response = completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            # api_base=API_BASE,
            api_key=GEMINI_API_KEY
        )
        result = response.choices[0].message.content
        if result is None:
            print("LLM returned None response")
            return "Error: LLM returned no content."
        print(f"LLM response: {result}")
        return result
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "Error: Unable to process the request at this time."

def contains_keywords(text: str, keywords: List[str]) -> bool:
    """
    Check if the text contains any of the provided keywords (case-insensitive).
    Used for filtering irrelevant search results.
    """
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

# =============================================================================
# Node Functions
# =============================================================================
def initialize_state_node(state: State) -> Dict[str, Any]:
    query = state.original_query
    print(f"Initializing state with query: {query}")
    return {
        "report": {"query": query, "results": [], "sources": [], "notes": ""},
        "current_query": query,
        "search_results": [],
        "step_count": 0,
        "max_steps": MAX_STEPS,
        "next_node": "query_refinement",
        "useful_databases": [],
        "consecutive_no_new_results": 0,
        "previous_queries": [],
        "domain_usage": {},
        "excluded_domains": [],
        "search_cache": {}
    }

def query_refinement_node(state: State) -> Dict[str, Any]:
    """
    Asks the LLM for a refined web search query. If it's identical to previous queries,
    we can detect that to avoid infinite loops.
    """
    current_query = state.current_query
    result_count = len(state.report.results)
    prompt = (
        f"You are assisting a researcher investigating adult content.\n"
        f"Given the current search query: '{current_query}', and the fact that {result_count} relevant results have been found so far, please suggest a refined search query.\n"
        "Include specific keywords such as 'scene', 'studio', or other relevant terms to yield more specific results.\n"
        "Return only a valid JSON object with the key 'refined_query'. Do not include any additional commentary."
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

def search_node(state: State) -> Dict[str, Any]:
    """
    Performs a general search for the current_query while excluding domains present in state.excluded_domains.
    Exclusion is done by appending '-site:' clauses to the search query.
    Uses search_cache to avoid redundant searches.
    """
    excluded_domains = state.excluded_domains
    query = state.current_query
    if excluded_domains:
        exclusion_clause = " ".join(f"-site:{domain}" for domain in excluded_domains)
        query_with_exclusion = f"{query} {exclusion_clause}"
    else:
        query_with_exclusion = query
    print(f"Searching with query: {query_with_exclusion}")
    search_cache = dict(state.search_cache)  # Copy existing cache
    if query_with_exclusion in search_cache:
        print(f"Using cached results for query: {query_with_exclusion}")
        results = search_cache[query_with_exclusion]
    else:
        try:
            results = search_tool.invoke(query_with_exclusion)
            print(f"Search results: {results}")
            search_cache[query_with_exclusion] = results
        except Exception as e:
            print(f"Error during search_tool.invoke for query '{query_with_exclusion}': {e}")
            results = []
    
    return {"search_results": results, "search_cache": search_cache}

def data_processing_node(state: State) -> Dict[str, Any]:
    """
    Takes the raw search results, filters irrelevant ones, calls the LLM to extract scene info,
    and appends them to the report if relevant. Also tracks domain usage.
    """
    report = state.report
    useful_databases = state.useful_databases
    domain_usage = dict(state.domain_usage)  # Local copy to update

    # Filter search results based on keywords from the original query
    keywords = state.original_query.lower().split()
    filtered_results = [
        res for res in state.search_results
        if contains_keywords(res.get('title', '') + ' ' + res.get('snippet', ''), keywords)
    ]
    print(f"Filtered {len(state.search_results) - len(filtered_results)} out of {len(state.search_results)} search results")
    print(f"Processing {len(filtered_results)} filtered search results")
    new_results_found = False

    for result in filtered_results:
        title = result.get('title', '')
        link = result.get('link', '')
        snippet = result.get('snippet', '')

        prompt = (
            f"You are analyzing a search result in relation to the query: '{state.original_query}'.\n\n"
            "Search result:\n"
            f"Title: {title}\n"
            f"Link: {link}\n"
            f"Snippet: {snippet}\n\n"
            "Determine if this result references a scene that matches the query.\n"
            "For example, if the query specifies a performer and genre, the scene should feature that performer and align with the genre.\n"
            "However, if the query explicitly mentions a type of fetish, genre, or action that is not part of the scene, it should not be a match.\n"
            "If it matches, extract the following details:\n"
            " - scene (scene name)\n"
            " - studio (studio or production company)\n"
            " - link (direct link to the scene)\n"
            " - description (brief description of the scene)\n"
            " - reason_for_inclusion (why this scene was included based on the query)\n"
            " - relevance (should be 'yes')\n"
            "If it does not match, return the following with null values:\n"
            " - relevance (should be 'no')\n"
            " - scene: null\n"
            " - studio: null\n"
            " - link: null\n"
            " - description: null\n"
            " - reason_for_inclusion: null\n"
            "Return only a valid JSON object with the keys: relevance, scene, studio, link, description, reason_for_inclusion. Do not include any additional text."
        )
        response = llm_call(prompt)
        extracted_data = extract_json(response)

        # Validate and process the extracted data
        if isinstance(extracted_data, dict) and extracted_data.get('relevance') == 'yes':
            scene_data = {
                "scene": extracted_data.get('scene', ''),
                "studio": extracted_data.get('studio', ''),
                "link": extracted_data.get('link', ''),
                "description": extracted_data.get('description', ''),
                "reason_for_inclusion": extracted_data.get('reason_for_inclusion', '')
            }
            # Ensure required fields are non-empty before adding to report
            if scene_data['scene'] and scene_data['link']:
                domain = urlparse(scene_data["link"]).netloc
                scene_data["verification"] = f"Source: {domain}, manual verification needed"

                # Check for duplicates before adding
                if not any(r.get("link") == scene_data["link"] for r in report.results):
                    report.results.append(scene_data)
                    new_results_found = True
                    print(f"Added scene: {scene_data['scene']} (Studio: {scene_data['studio']}, Link: {scene_data['link']})")

                    if domain and domain not in report.sources:
                        report.sources.append(domain)
                        print(f"Added source: {domain}")

                    if domain and domain not in useful_databases:
                        useful_databases.append(domain)

                if domain:
                    domain_usage[domain] = domain_usage.get(domain, 0) + 1

    # Update consecutive count based on whether new results were found
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
        "consecutive_no_new_results": consecutive_count,
        "domain_usage": domain_usage
    }

def review_node(state: State) -> Dict[str, Any]:
    """
    Reviews the domains used so far. If there's excessive usage of any domain,
    the LLM is asked to reflect on whether we should exclude it. Also uses the LLM 
    to see if we're missing any important domains. The updated excluded domains list is returned.
    """
    domain_usage = state.domain_usage
    overused_domains = [dom for dom, count in domain_usage.items() if count >= USAGE_THRESHOLD]
    new_excluded = set(state.excluded_domains).union(overused_domains)

    coverage_prompt = (
        "We have collected adult scene data from various domains.\n"
        f"Current domain usage:\n{json.dumps(domain_usage, indent=2)}\n\n"
        f"Domains excluded so far: {state.excluded_domains}\n"
        "Please analyze if there are any important domains missing or if certain domains have been overused.\n"
        "Return only a valid JSON object with the key 'rationale' that provides a brief explanation or next-step advice. Do not include any additional commentary."
    )
    coverage_response = llm_call(coverage_prompt)
    coverage_json = extract_json(coverage_response)
    rationale = coverage_json.get("rationale") if isinstance(coverage_json, dict) else None
    if rationale:
        print(f"LLM Rationale: {rationale}")
    else:
        print("No special rationale from LLM or invalid JSON returned.")

    return {
        "excluded_domains": list(new_excluded)
    }

def quality_check_node(state: State) -> Dict[str, Any]:
    """
    Checks if we have enough results or if we've tried too many times.
    If we haven't found anything new for N times, we also stop early.
    """
    state.step_count += 1
    current_results = len(state.report.results)
    if (
        state.step_count >= state.max_steps
        or current_results >= TARGET_RESULTS
        or state.consecutive_no_new_results >= NO_RESULT_LIMIT
    ):
        next_node = "response"
        if state.consecutive_no_new_results >= NO_RESULT_LIMIT and current_results == 0:
            state.report.notes = "No relevant results found after multiple refinements."
    else:
        next_node = "query_refinement"

    print(f"Step {state.step_count}/{state.max_steps}: {current_results} results found. Next node: {next_node}")
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    """
    Final node: outputs the final aggregated report, generates a Markdown report,
    and saves it to the "reports" folder.
    """
    print("Generating final report")
    report = state.report

    # Generate Markdown report with new fields
    md_content = f"# Report for Query: {report.query}\n\n"
    md_content += "## Results\n"
    for result in report.results:
        md_content += f"- **Scene:** {result['scene']}\n"
        md_content += f"  - **Studio:** {result['studio']}\n"
        md_content += f"  - **Link:** {result['link']}\n"
        md_content += f"  - **Description:** {result.get('description', 'N/A')}\n"
        md_content += f"  - **Reason for Inclusion:** {result.get('reason_for_inclusion', 'N/A')}\n"
        md_content += f"  - **Verification:** {result['verification']}\n\n"
    md_content += "## Sources\n"
    for source in report.sources:
        md_content += f"- {source}\n"
    md_content += "\n## Notes\n"
    md_content += report.notes if report.notes else "No additional notes."

    # Ensure reports folder exists
    os.makedirs("reports", exist_ok=True)

    # Generate filename
    query_slug = re.sub(r'[^\w\-_\.]', '_', report.query)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/report_{query_slug}_{timestamp}.md"

    # Write to file with error handling
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Error saving report: {e}")

    print(f"Final report - Query: {report.query}, Results: {len(report.results)}, Sources: {len(report.sources)}")
    return {"report": report}

# =============================================================================
# Graph Definition
# =============================================================================
from langgraph.pregel import GraphRecursionError

workflow = StateGraph(State)
workflow.add_node("initialize", initialize_state_node)
workflow.add_node("query_refinement", query_refinement_node)
workflow.add_node("search", search_node)
workflow.add_node("data_processing", data_processing_node)
workflow.add_node("review_node", review_node)
workflow.add_node("quality_check", quality_check_node)
workflow.add_node("response", response_node)

workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "query_refinement")
# Removed the database_selection node; now directly transition from query_refinement to search.
workflow.add_edge("query_refinement", "search")
workflow.add_edge("search", "data_processing")
workflow.add_edge("data_processing", "review_node")
workflow.add_edge("review_node", "quality_check")

workflow.add_conditional_edges(
    "quality_check",
    lambda state: state.next_node,
    {"query_refinement": "query_refinement", "response": "response"}
)

workflow.add_edge("response", END)
graph = workflow.compile()

# =============================================================================
# Main Agent Function
# =============================================================================
def run_agent(query: str) -> Dict[str, Any]:
    print(f"Starting agent with query: {query}")
    initial_state = State(original_query=query)
    final_data = graph.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})
    final_state = State(**dict(final_data))
    print("Agent execution completed")
    return final_state.report

if __name__ == "__main__":
    report = run_agent(QUERY)
    print("\n=== Final Report ===")
    print(f"Query: {report.query}")
    print("Results:")
    for result in report.results:
        print(f"- {result['scene']} (Studio: {result['studio']}, Link: {result['link']}, Verification: {result['verification']})")
    print(f"Sources: {report.sources}")
    print(f"Notes: {report.notes}")
