import os
import re
import json
import time
import pathlib
import requests  # For fetching page content during follow-up browsing
from bs4 import BeautifulSoup  # For parsing HTML content
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, ValidationError

from langchain_community.tools import DuckDuckGoSearchResults
from litellm import completion
from langgraph.graph import StateGraph, START, END

###############################################################################
# Load environment variables
###############################################################################
load_dotenv()  # Loads .env if present; environment variables override

###############################################################################
# Configuration (read from ENV, with defaults)
###############################################################################
MODEL = os.getenv("MODEL", "ollama/qwen2.5-coder")
API_BASE = os.getenv("API_BASE", "http://localhost:11434")

# Maximum node transitions
MAX_STEPS = int(os.getenv("MAX_STEPS", 20))

# (No longer used; concurrency removed)
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", 3))

# How many results to process per chunk
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 25))

# Results from each DuckDuckGo search
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 25))

# Maximum follow-up browsing depth
BROWSE_DEPTH = int(os.getenv("BROWSE_DEPTH", 1))

# Thresholds and stopping conditions
USAGE_THRESHOLD = int(os.getenv("USAGE_THRESHOLD", 3))   # If domain usage > threshold, consider excluding
TARGET_RESULTS = int(os.getenv("TARGET_RESULTS", 3))     # If we find >= this many results, we can stop
NO_RESULT_LIMIT = int(os.getenv("NO_RESULT_LIMIT", 3))   # If we fail to find new results for these many consecutive tries, stop

# How many refined queries to generate from the initial research
NUM_SUBQUERIES = int(os.getenv("NUM_SUBQUERIES", 3))

###############################################################################
# Pydantic models
###############################################################################

class Report(BaseModel):
    # Updated to match the new requirement structure
    original_query: str = ""
    topic_of_research: str = ""  # e.g. short "topic" for the entire search
    # Each result must contain the fields below (with optional expansions)
    # We'll store them as dictionaries for maximum flexibility,
    # but we enforce the keys: scene, studio, link, cast, tags, reason
    results: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    notes: str = ""

class State(BaseModel):
    report: Report = Field(default_factory=Report)

    # The "main" or current query for each cycle of searching
    current_query: str = ""

    # We will hold multiple queries to search on in each loop:
    generated_queries: List[str] = Field(default_factory=list)

    databases: List[Dict[str, str]] = Field(default_factory=list)
    search_results: List[Dict[str, str]] = Field(default_factory=list)

    step_count: int = 0
    max_steps: int = MAX_STEPS

    next_node: str = ""
    original_query: str = ""

    useful_databases: List[str] = Field(default_factory=list)
    consecutive_no_new_results: int = 0
    previous_queries: List[str] = Field(default_factory=list)
    domain_usage: Dict[str, int] = Field(default_factory=dict)
    excluded_domains: List[str] = Field(default_factory=list)
    browse_depth: int = 0
    max_browse_depth: int = BROWSE_DEPTH

    class Config:
        extra = "allow"

class PreliminaryQueries(BaseModel):
    subqueries: List[str]

class SearchQuery(BaseModel):
    search_query: str

# Modified: "studio" is optional (default None), additional optional fields
# for cast, tags, reason (why accepted)
class SceneInfo(BaseModel):
    scene: str
    studio: Optional[str] = None
    link: str
    cast: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    reason: Optional[str] = None

# For deduplication
class DeduplicatedScene(BaseModel):
    scene: str
    studio: str
    links: List[str]
    cast: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    reason: Optional[str] = None

###############################################################################
# Helper Functions
###############################################################################

def extract_json(response: str):
    """
    Extract JSON from a response string.
    Removes markdown code fences if present.
    """
    response = response.strip()
    if response.startswith("```"):
        # Remove starting and ending code fences if present
        response = response.strip("```")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

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

def save_report_to_md(report: Report) -> str:
    """
    Save the final report to a markdown file, reflecting the updated structure:
    - Original Query
    - Topic of Research
    - Results with Scene, Studio, Link, Cast, Tags, Reason
    - Sources
    - Notes
    """
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    file_path = reports_dir / filename

    md_content = [
        f"# Original Query: {report.original_query}\n\n",
        f"## Topic of Research: {report.topic_of_research}\n\n",
        "## Results\n\n"
    ]

    if report.results:
        for idx, result in enumerate(report.results, start=1):
            scene = result.get("scene", "")
            studio = result.get("studio", "")
            link = result.get("link", None)
            # If aggregated links exist:
            links = result.get("links", [])
            cast = result.get("cast", [])
            tags = result.get("tags", [])
            reason = result.get("reason", "")

            md_content.append(f"**Result {idx}:**\n")
            md_content.append(f"- Scene Name: {scene}\n")
            md_content.append(f"- Studio: {studio}\n")

            # We prefer "links" if present, else fall back to single link
            if links and isinstance(links, list) and len(links) > 0:
                md_content.append("- Links:\n")
                for lidx, l in enumerate(links, start=1):
                    md_content.append(f"  {lidx}. {l}\n")
            elif link:
                md_content.append(f"- Link: {link}\n")

            if cast:
                md_content.append(f"- Cast: {', '.join(cast)}\n")
            if tags:
                md_content.append(f"- Tags: {', '.join(tags)}\n")

            md_content.append(f"- Reason why accepted: {reason}\n\n")

    md_content.append("## Sources\n")
    for src in report.sources:
        md_content.append(f"- {src}\n")

    md_content.append("\n## Notes\n")
    md_content.append(f"{report.notes}\n\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("".join(md_content))

    print(f"Report saved to {file_path}")
    return str(file_path)

###############################################################################
# Node Functions
###############################################################################

def initialize_state_node(state: State) -> Dict[str, Any]:
    """
    Start here. We'll store the original query in the report,
    and also produce a short "topic_of_research" from the LLM
    so we can place it in the final report as well.
    """
    query = state.original_query
    print(f"Initializing state with query: {query}")

    # Let the LLM produce a short "topic" label
    prompt_topic = (
        f"Given the user query: '{query}', please provide a short phrase or label "
        "that describes the topic of research.\n"
        "Return JSON with key 'topic_of_research'. Example:\n"
        "{\n"
        "   \"topic_of_research\": \"Adult scenes featuring performer X\"\n"
        "}"
    )
    topic_response = llm_call(prompt_topic)
    extracted_topic = extract_json(topic_response)
    topic_str = ""
    if isinstance(extracted_topic, dict):
        topic_str = extracted_topic.get("topic_of_research", "")

    return {
        "report": {
            "original_query": query,
            "topic_of_research": topic_str,
            "results": [],
            "sources": [],
            "notes": ""
        },
        "current_query": query,
        "generated_queries": [],
        "databases": [],
        "search_results": [],
        "step_count": 0,
        "max_steps": MAX_STEPS,
        "next_node": "preliminary_research",
        "useful_databases": [],
        "consecutive_no_new_results": 0,
        "previous_queries": [],
        "domain_usage": {},
        "excluded_domains": [],
        "browse_depth": 0
    }

def preliminary_research_node(state: State) -> Dict[str, Any]:
    """
    Gather background info with an initial pass and produce multiple refined subqueries.
    We'll generate a handful of queries (NUM_SUBQUERIES) that we can run.
    """
    user_query = state.current_query

    # First, do a broad search to gather background context
    print(f"Performing initial broad search for context on: {user_query}")
    try:
        results = search_tool.invoke(user_query)
    except Exception as e:
        print(f"Error searching '{user_query}': {e}")
        results = []

    # We'll glean a big snippet for context
    background_snippets = []
    for r in results[:30]:
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        if snippet:
            background_snippets.append(f"- {snippet} (Link: {link})")

    background_text = "\n".join(background_snippets)

    # Now ask the LLM to produce N refined queries based on the background
    prompt_subqueries = (
        f"You have the original user query:\n"
        f"\"{user_query}\"\n\n"
        "Below is some background info from an initial broad search:\n"
        f"{background_text}\n\n"
        f"Generate {NUM_SUBQUERIES} refined or related search queries that would better capture the relevant results.\n\n"
        "Return JSON with key 'subqueries' as an array of strings.\n"
        "Example:\n"
        "{\n"
        "  \"subqueries\": [\"Refined Query 1\", \"Refined Query 2\", \"Refined Query 3\"]\n"
        "}"
    )
    response_subqueries = llm_call(prompt_subqueries)
    extracted_subqueries_json = extract_json(response_subqueries)
    subqueries_list = []
    if isinstance(extracted_subqueries_json, dict):
        try:
            subq_obj = PreliminaryQueries(**extracted_subqueries_json)
            subqueries_list = subq_obj.subqueries
        except ValidationError as e:
            print(f"Validation error in subqueries: {e}")
    if not subqueries_list:
        # fallback
        subqueries_list = [f"{user_query} additional info"]

    print(f"Refined subqueries:\n{subqueries_list}")

    # We'll store these subqueries in the state for the next node to search
    return {
        "generated_queries": subqueries_list
    }

def database_selection_node(state: State) -> Dict[str, Any]:
    """
    We pick which domains we want to search across. We'll do a quick pass
    on the first refined query from `state.generated_queries` just to
    detect candidate domains. Then LLM decides which domains to keep.
    """
    generated_queries = state.generated_queries
    if not generated_queries:
        # fallback
        generated_queries = [state.current_query]

    # We'll pick the first refined query to do domain detection
    first_refined_query = generated_queries[0]
    print(f"Detecting candidate domains with query: {first_refined_query}")

    try:
        results = search_tool.invoke(first_refined_query)
        print(f"Search results for domain detection: {results}")
    except Exception as e:
        print(f"Error during search_tool.invoke: {e}")
        results = []

    domains = []
    for result in results:
        url = result.get('link', '')
        domain = urlparse(url).netloc
        if domain and domain not in domains:
            domains.append(domain)

    if not domains:
        print("No domains discovered. Using only a general search fallback.")
        return {"databases": [{"type": "general"}]}

    # We let the LLM decide which domains to keep
    domain_verification_prompt = (
        f"You are verifying potential sources for the user query:\n"
        f"\"{state.original_query}\"\n\n"
        "We discovered these candidate domains:\n"
        f"{json.dumps(domains, indent=2)}\n\n"
        "Please output JSON with a key 'keep_domains', which is an array of domains "
        "that are likely to have relevant info about the query, and optionally a 'reason' key for your rationale.\n"
        "If you don't think we need any domain, you may provide an empty array."
    )
    domain_verification_response = llm_call(domain_verification_prompt)
    domain_verification_json = extract_json(domain_verification_response)

    keep_domains = domains
    if isinstance(domain_verification_json, dict):
        suggested_keeps = domain_verification_json.get("keep_domains")
        if isinstance(suggested_keeps, list) and suggested_keeps:
            keep_domains = [d for d in suggested_keeps if d in domains]

    print(f"Domains the LLM chose to keep: {keep_domains}")

    if not keep_domains:
        print("LLM returned no relevant domains; using only a general fallback.")
        return {"databases": [{"type": "general"}]}

    # Exclude domains that are already in excluded_domains
    keep_filtered = [d for d in keep_domains if d not in state.excluded_domains]

    if not keep_filtered:
        print("All chosen domains were excluded, so only using general fallback.")
        return {"databases": [{"type": "general"}]}

    # We store them each as site plus a general fallback
    dbs = [{"type": "site", "url": d} for d in keep_filtered]
    dbs.append({"type": "general"})
    print(f"Selected databases: {dbs}")

    return {"databases": dbs}

def search_node(state: State) -> Dict[str, Any]:
    """
    Search across each "generated_query" for each database, combining the results.
    We skip domains in state.excluded_domains if they're site-based.
    """
    combined_results = []
    excluded_domains = set(state.excluded_domains)

    # We'll search for each refined query
    for sub_query in state.generated_queries:
        for db in state.databases:
            if db["type"] == "site":
                if db["url"] in excluded_domains:
                    print(f"Skipping domain {db['url']} because it is excluded.")
                    continue
                query = f"site:{db['url']} {sub_query}"
            else:
                query = sub_query

            print(f"Searching with query: {query}")
            try:
                results = search_tool.invoke(query)
                combined_results.extend(results)
            except Exception as e:
                print(f"Error searching '{query}': {e}")

    print(f"Collected {len(combined_results)} results from all refined queries+databases.")
    return {"search_results": combined_results}

def follow_up_node(state: State) -> Dict[str, Any]:
    """
    If requested, attempt to do a deeper "browse" of certain links to glean
    more structured details. We skip if max_browse_depth reached.
    """
    if state.browse_depth >= state.max_browse_depth:
        print("Maximum browse depth reached, skipping follow-up browsing.")
        return {}
    if not state.search_results:
        print("No search results available for follow-up browsing.")
        return {}

    new_browse_results = []
    for i in range(0, len(state.search_results), CHUNK_SIZE):
        chunk = state.search_results[i: i + CHUNK_SIZE]
        chunk_prompt_lines = []
        for idx, res in enumerate(chunk):
            chunk_prompt_lines.append(
                f"{idx+1}. Title: {res.get('title','')}\n   Link: {res.get('link','')}\n   Snippet: {res.get('snippet','')}"
            )
        chunk_text = "\n".join(chunk_prompt_lines)
        prompt = (
            f"You are reviewing the following search results for further browsing to gather more detailed information relevant to the user query:\n"
            f"'{state.current_query}'\n\n"
            f"{chunk_text}\n\n"
            "For any result that you think warrants additional browsing (i.e. fetching and analyzing the webpage content for more details), "
            "return a JSON object with a key 'indices_to_browse' as an array of 1-based indices indicating which results to browse. "
            "If none, return an empty array.\n"
            "Example:\n"
            "{\n"
            "  \"indices_to_browse\": [1, 3]\n"
            "}"
        )
        response = llm_call(prompt)
        extracted = extract_json(response)
        indices = []
        if isinstance(extracted, dict):
            indices = extracted.get("indices_to_browse", [])
        for index in indices:
            idx = index - 1  # convert 1-based to 0-based index
            if idx < 0 or idx >= len(chunk):
                continue
            result = chunk[idx]
            url = result.get("link")
            if not url:
                continue
            try:
                print(f"Fetching content from {url} for follow-up browsing.")
                # Increase timeout to 30 seconds and add retries
                for attempt in range(3):
                    try:
                        resp = requests.get(url, timeout=30)
                        if resp.status_code == 200:
                            break
                    except requests.exceptions.RequestException as e:
                        print(f"Attempt {attempt+1} failed: {e}")
                        if attempt == 2:
                            raise
                soup = BeautifulSoup(resp.text, 'html.parser')
                for tag in soup(["script", "style"]):
                    tag.decompose()
                page_text = soup.get_text(separator=" ", strip=True)
                # Truncate page_text if too long
                if len(page_text) > 10000:
                    page_text = page_text[:10000] + "... [truncated]"
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue

            # Prompt LLM to parse the page for relevant structured info
            browse_prompt = (
                f"You are browsing to gather more detailed information relevant to the user query: '{state.current_query}'.\n"
                f"Here is the extracted text content from the page:\n"
                f"{page_text}\n\n"
                "Extract any additional structured data related to the query. For each relevant piece of information, "
                "return an object with keys 'scene', 'studio', 'link', 'cast', 'tags', and 'reason' (the reason is why you accepted it). "
                "The 'link' key should be the original URL.\n"
                "Return a JSON array of such objects. If nothing is relevant, return an empty JSON array."
            )
            browse_response = llm_call(browse_prompt)
            extracted_info = extract_json(browse_response)
            if isinstance(extracted_info, list):
                for item in extracted_info:
                    try:
                        scene_info = SceneInfo(**item)
                        # We'll store it as a dict, possibly partial
                        new_browse_results.append({
                            "scene": scene_info.scene,
                            "studio": scene_info.studio,
                            "link": scene_info.link,
                            "cast": scene_info.cast,
                            "tags": scene_info.tags,
                            "reason": scene_info.reason,
                            "verification": f"Browsed from {url}"
                        })
                    except ValidationError as e:
                        print(f"Validation error in follow-up browsing extraction: {e}")

    # Add these new results to the final report
    for item in new_browse_results:
        # Prevent duplicates by link
        if not any(r.get("link") == item.get("link") for r in state.report.results):
            state.report.results.append(item)
            domain = urlparse(item["link"]).netloc
            if domain and domain not in state.report.sources:
                state.report.sources.append(domain)
    state.browse_depth += 1
    print(f"Follow-up browsing completed. Updated results count: {len(state.report.results)}. Browse depth is now {state.browse_depth}.")
    return {"report": state.report}

def data_processing_node(state: State) -> Dict[str, Any]:
    """
    Vet and parse the newly returned search_results. For each chunk, ask the LLM
    to check if it is relevant to the original query, and if so to extract
    structured data: scene, studio, link, cast, tags, reason.
    """
    report = state.report
    useful_databases = state.useful_databases
    search_results = state.search_results
    domain_usage = dict(state.domain_usage)
    original_query = state.original_query

    print(f"Processing {len(search_results)} search results...")
    new_results_found = False

    if not search_results:
        print("No new search results to process.")
        consecutive_count = state.consecutive_no_new_results + 1
        return {
            "report": report,
            "search_results": [],
            "useful_databases": useful_databases,
            "consecutive_no_new_results": consecutive_count,
            "domain_usage": domain_usage
        }

    def process_chunk(chunk_data: List[Dict[str, Any]]):
        chunk_prompt_lines = []
        for idx, res in enumerate(chunk_data):
            chunk_prompt_lines.append(
                f"{idx+1}. Title: {res.get('title','')}\n   Link: {res.get('link','')}\n   Snippet: {res.get('snippet','')}"
            )
        results_text = "\n".join(chunk_prompt_lines)
        prompt = (
            f"You are extracting structured data based on the user query:\n"
            f"'{original_query}'\n\n"
            "Below are some search results. If the result is NOT relevant to the user's exact query, omit it.\n"
            "For each truly relevant result, extract these fields:\n"
            "- scene (string)\n"
            "- studio (string, optional)\n"
            "- link (string)\n"
            "- cast (array of names, optional)\n"
            "- tags (array of strings, optional)\n"
            "- reason (string, a short explanation of why you accepted it)\n\n"
            f"{results_text}\n\n"
            "Return a JSON array of objects. Example:\n"
            "[\n"
            "  {\n"
            "    \"scene\": \"...\",\n"
            "    \"studio\": \"...\",\n"
            "    \"link\": \"...\",\n"
            "    \"cast\": [\"Performer1\", \"Performer2\"],\n"
            "    \"tags\": [\"tag1\", \"tag2\"],\n"
            "    \"reason\": \"snippet references the exact query...\"\n"
            "  }\n"
            "]"
        )
        response = llm_call(prompt)
        return extract_json(response) or []

    chunked_results = []
    for i in range(0, len(search_results), CHUNK_SIZE):
        chunk = search_results[i : i + CHUNK_SIZE]
        result = process_chunk(chunk)
        chunked_results.extend(result)

    for scene_data in chunked_results:
        if not isinstance(scene_data, dict):
            print(f"Skipping non-dict item in chunked results: {scene_data}")
            continue
        try:
            scene_info = SceneInfo(**scene_data)
        except Exception as e:
            print(f"Error processing scene_data: {e} with data: {scene_data}")
            continue

        # Build the final dictionary
        scene = {
            "scene": scene_info.scene,
            "studio": scene_info.studio if scene_info.studio else "",
            "link": scene_info.link,
            "cast": scene_info.cast if scene_info.cast else [],
            "tags": scene_info.tags if scene_info.tags else [],
            "reason": scene_info.reason if scene_info.reason else "",
        }
        domain = urlparse(scene["link"]).netloc
        scene["verification"] = f"Source: {domain}, manual verification needed"

        # De-duplicate by link
        if not any(r.get("link") == scene["link"] for r in report.results):
            report.results.append(scene)
            new_results_found = True
            print(f"Added scene/item: {scene['scene']} (Studio: {scene['studio']}, Link: {scene['link']})")
            if domain and domain not in report.sources:
                report.sources.append(domain)
                print(f"Added source: {domain}")
            if domain and domain not in useful_databases:
                useful_databases.append(domain)
        # track domain usage
        if domain:
            domain_usage[domain] = domain_usage.get(domain, 0) + 1

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
    Check domain usage. If usage is above threshold, mark as excluded
    to ensure we get diverse results in subsequent loops.
    """
    domain_usage = state.domain_usage
    overused_domains = [dom for dom, cnt in domain_usage.items() if cnt >= USAGE_THRESHOLD]
    new_excluded = set(state.excluded_domains).union(overused_domains)

    coverage_prompt = (
        "We have collected data from various domains.\n"
        f"Current domain usage:\n{json.dumps(domain_usage, indent=2)}\n\n"
        f"Domains excluded so far: {state.excluded_domains}\n"
        "Reflect if we are missing important domains or if we have overused certain domains.\n"
        "Provide a short rationale in JSON under key 'rationale'. Example:\n"
        "{\n"
        "  \"rationale\": \"We might want to exclude repeated domains for more variety.\"\n"
        "}"
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

def deduplicate_node(state: State) -> Dict[str, Any]:
    """
    Attempt to unify duplicates (same scene, multiple links).
    We'll ask the LLM to unify them into objects that contain:
    scene, studio, links[], cast, tags, reason
    """
    current_results = state.report.results
    if not current_results:
        print("No results to deduplicate.")
        return {"report": state.report.dict()}

    results_json = json.dumps(current_results, indent=2)
    prompt = (
        "We have the following JSON array of data items (some may be duplicates). "
        "Each item has keys: scene, studio, link, cast, tags, reason, verification.\n"
        "If multiple entries refer to the same exact scene, unify them into a single object.\n"
        "The unified object should have:\n"
        "  - scene (string)\n"
        "  - studio (string)\n"
        "  - links (array of distinct links)\n"
        "  - cast (array of distinct names if known)\n"
        "  - tags (array of distinct tags)\n"
        "  - reason (string, can unify or concatenate reasons)\n"
        "Return a JSON array of deduplicated objects, each with at least [scene, studio, links].\n"
        "If there's no duplication, simply convert single link -> 'links'.\n\n"
        f"Current results:\n```json\n{results_json}\n```"
    )
    response = llm_call(prompt)
    extracted = extract_json(response)
    if not isinstance(extracted, list):
        print("No valid deduplication JSON returned, skipping deduplication.")
        return {"report": state.report.dict()}

    deduplicated_items = []
    for item in extracted:
        try:
            deduped = DeduplicatedScene(**item)
            # We'll store it with optional cast/tags
            deduplicated_items.append({
                "scene": deduped.scene,
                "studio": deduped.studio,
                "links": deduped.links,
                "cast": deduped.cast if deduped.cast else [],
                "tags": deduped.tags if deduped.tags else [],
                "reason": deduped.reason if deduped.reason else ""
            })
        except ValidationError as e:
            print(f"Validation error in deduplicated item: {e}")
            continue

    print(f"Deduplicated {len(current_results)} items down to {len(deduplicated_items)} items.")
    # Update the state.report in place
    state.report.results = deduplicated_items
    return {"report": state.report.dict()}

def quality_check_node(state: State) -> Dict[str, Any]:
    """
    Decide if we continue or finalize:
    - If step_count >= max_steps, or
      current results >= TARGET_RESULTS, or
      consecutive_no_new_results >= NO_RESULT_LIMIT
      => proceed to response
    Else => go back to database_selection (loop).
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
        next_node = "database_selection"

    print(f"Step {state.step_count}/{state.max_steps}: {current_results} results found. Next node: {next_node}")
    state.next_node = next_node
    return {"step_count": state.step_count, "next_node": next_node}

def response_node(state: State) -> Dict[str, Any]:
    """
    Generate final output, save to MD, end.
    """
    print("Generating final report.")
    report = state.report
    print(
        f"Final report - Original Query: {report.original_query}, "
        f"Topic: {report.topic_of_research}, "
        f"Results: {len(report.results)}, "
        f"Sources: {len(report.sources)}"
    )
    save_report_to_md(report)
    return {"report": report}

###############################################################################
# Graph Definition
###############################################################################

search_tool = DuckDuckGoSearchResults(output_format="list", max_results=MAX_SEARCH_RESULTS)

workflow = StateGraph(State)

workflow.add_node("initialize", initialize_state_node)
workflow.add_node("preliminary_research", preliminary_research_node)
workflow.add_node("database_selection", database_selection_node)
workflow.add_node("search", search_node)
workflow.add_node("follow_up", follow_up_node)  # If we want deeper browsing
workflow.add_node("data_processing", data_processing_node)
workflow.add_node("review_node", review_node)
workflow.add_node("deduplicate_node", deduplicate_node)
workflow.add_node("quality_check", quality_check_node)
workflow.add_node("response", response_node)

# Execution path
workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "preliminary_research")
workflow.add_edge("preliminary_research", "database_selection")
workflow.add_edge("database_selection", "search")
workflow.add_edge("search", "follow_up")
workflow.add_edge("follow_up", "data_processing")
workflow.add_edge("data_processing", "review_node")
workflow.add_edge("review_node", "deduplicate_node")
workflow.add_edge("deduplicate_node", "quality_check")

workflow.add_conditional_edges(
    "quality_check",
    lambda state: state.next_node,
    {
        "database_selection": "database_selection",
        "response": "response"
    }
)

workflow.add_edge("response", END)
graph = workflow.compile()

###############################################################################
# Main Agent Function
###############################################################################

def run_agent(query: str) -> Dict[str, Any]:
    """
    Main entry point to execute the entire workflow for a given query.
    """
    print(f"Starting agent with query: {query}")
    initial_state = State(original_query=query)
    final_data = graph.invoke(initial_state)
    final_state = State(**dict(final_data))
    print("Agent execution completed.")
    return final_state.report

if __name__ == "__main__":
    # Example usage
    sample_query = "Rebecca Linares Anal Scenes"
    report = run_agent(sample_query)
    print("\n=== Final Report ===")
    print(f"Original Query: {report.original_query}")
    print(f"Topic of Research: {report.topic_of_research}")
    print("Results:")
    for idx, result in enumerate(report.results, start=1):
        scene = result.get("scene")
        studio = result.get("studio")
        links = result.get("links", [])
        if not links:
            link = result.get("link")
            links = [link] if link else []
        cast = result.get("cast", [])
        tags = result.get("tags", [])
        reason = result.get("reason", "")
        print(f"{idx}. Scene: {scene}, Studio: {studio}")
        print(f"   Links: {links}")
        print(f"   Cast: {cast}")
        print(f"   Tags: {tags}")
        print(f"   Reason: {reason}")
    print(f"Sources: {report.sources}")
    print(f"Notes: {report.notes}")