"""
Agentic RAG — Root Agent.

PURPOSE:
  Single LlmAgent that implements the full Agentic RAG pipeline internally.
  Uses a step-by-step instruction to: analyze query, search RAG store,
  evaluate results, search web if needed, then synthesize answer.

ARCHITECTURE:
  Single agent with all tools — avoids the 5x API call overhead of
  SequentialAgent while achieving the same Agentic RAG behavior.
"""

try:
    from .config import MODEL_NAME
    from .tools.rag_tool import rag_search
    from .tools.web_search_tool import web_search
except ImportError:
    from config import MODEL_NAME
    from tools.rag_tool import rag_search
    from tools.web_search_tool import web_search

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# ── Wrap async tools as FunctionTool instances ──────────────────────

rag_search_tool = FunctionTool(
    func=rag_search,
)

web_search_tool = FunctionTool(
    func=web_search,
)

# ── Main Agentic RAG instruction ───────────────────────────────────

AGENTIC_RAG_INSTRUCTION = """\
You are an Agentic RAG system with 5 stages of processing. Execute them IN ORDER.

## Stage 1: Query Analysis
Analyze the user's question deeply. Identify:
- Core entities and relationships
- What type of information is needed (facts, procedures, explanations)
- The best search queries (formulate 1-2 specific queries)

## Stage 2: RAG Search
Call `rag_search(query=...)` for each formulated query.
If the first query gets high-quality results (score > 0.8), you may skip additional queries.
Read the returned passages carefully — they contain specific project details.

## Stage 3: Result Evaluation
Evaluate the RAG results:
- Do they directly answer the user's question?
- What specific details do they provide?
- What gaps remain that need supplementary information?

If the RAG results are sufficient, skip web search.
If there are clear gaps, proceed to Stage 4.

## Stage 4: Web Search (if needed)
Call `web_search(query=...)` to fill knowledge gaps.
Prefer project-specific web searches (documentation, technical references).

## Stage 5: Answer Synthesis
Combine everything into a comprehensive, well-structured answer.
- Start with a direct answer to the question
- Cite sources: mark RAG-sourced facts with [RAG: doc_name] and web-sourced facts with [Web: site]
- If the information is incomplete, say so clearly with appropriate confidence level
- Use markdown formatting (bullet points, code if relevant)
- Be specific and cite actual details from the retrieved documents

## Critical Rules
- ALWAYS call rag_search at least once before answering
- ONLY call web_search if RAG results have gaps
- Report an empty result gracefully: "No relevant information found in the document store"
- Do NOT fabricate or hallucinate information
- If a tool call fails, report: "Search failed, proceeding with available information"
- Keep track of which facts came from which source for proper citation
- Your answer is the FINAL output — be thorough and well-structured

Begin.
"""

# ── Build the agent ────────────────────────────────────────────────

agent = LlmAgent(
    name="AgenticRag",
    model=MODEL_NAME,
    description=(
        "Agentic RAG system that searches documents and the web, "
        "evaluates results, and synthesizes grounded answers."
    ),
    instruction=AGENTIC_RAG_INSTRUCTION,
    tools=[rag_search_tool, web_search_tool],
    output_key="final_answer",
)
