"""
Web Researcher Agent — Step 4 of the Agentic RAG pipeline.

PURPOSE:
  Searches the web for supplementary information to fill gaps identified
  by the Query Refiner. Uses the web_search tool (DuckDuckGo + Google News RSS).

BEHAVIOR:
  - Reads the Query Refiner's gap analysis and refined queries
  - Calls web_search for each gap query
  - Reports results clearly for the Answer Synthesizer

TOOLS:
  - web_search(query, max_results) — searches the web

AGENT NOTES:
  - No API key needed — uses DuckDuckGo with Google News RSS fallback.
  - Results include title, snippet, URL, and source.
"""

try:
    from ..tools.web_search_tool import web_search
except ImportError:
    from tools.web_search_tool import web_search

from google.adk.agents import LlmAgent

WEB_RESEARCHER_INSTRUCTION = """\
You are the Web Researcher — the fourth stage of an Agentic RAG pipeline.

YOUR JOB:
  Search the web for current information that fills the gaps identified
  by the Query Refiner. If the Query Refiner said results are sufficient,
  report that no web search is needed.

STEPS:
  1. Read the conversation history — find the Query Refiner's verdict:
     - If VERDICT is "proceed" → no web search needed.
       Just summarize and pass to the next agent.
     - If VERDICT is "refine_before_answer" or "supplement_with_web" →
       Extract the REFINED_QUERY or TARGETED_QUERY and search the web.
  2. Call `web_search(query=..., max_results=5)` for each identified query.
  3. Combine results. Deduplicate similar findings.
  4. Report clearly: what you searched, what you found, source URLs.

RULES:
  - ALWAYS call web_search — do NOT answer from your training data.
  - If the query is very specific, try 2-3 different phrasings.
  - Report source URLs so the Answer Synthesizer can cite them.
  - If search fails, say so and move on — the answer will use RAG only.
  - Your output will be read by the final Answer Synthesizer.

Start now.
"""

web_researcher_agent = LlmAgent(
    name="WebResearcher",
    model="gemini-2.0-flash",
    description=(
        "Searches the web for supplementary information "
        "to fill gaps in the RAG results."
    ),
    instruction=WEB_RESEARCHER_INSTRUCTION,
    tools=[web_search],
)
