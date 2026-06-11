"""
Query Analyzer Agent — Step 1 of the Agentic RAG pipeline.

PURPOSE:
  Takes the user's raw question and reformulates it into targeted
  search queries optimized for the RAG document store. Also identifies
  the user's intent and any key entities.

BEHAVIOR:
  - Reads the user's original question from session state['user_query']
  - Produces 1-3 reformulated search queries
  - Identifies the domain/topic of the question
  - Writes to session state: refined_queries, intent, entities

INPUT:
  User provides the question as the conversation turn text.
  The agent reads state['user_query'] (set by the SequentialAgent callback).

OUTPUT (writes to state):
  state['refined_queries'] = [str, ...]  — optimized search queries
  state['original_query'] = str           — the user's original question
  state['analysis'] = str                 — brief domain/intent analysis
"""

from google.adk.agents import LlmAgent

QUERY_ANALYZER_INSTRUCTION = """\
You are the Query Analyzer — the first stage of an Agentic RAG pipeline.

YOUR JOB:
  Analyze the user's question and reformulate it into precise search queries
  that will retrieve the most relevant information from the document store.

RULES:
  1. Read the user's question carefully.
  2. If the question is vague or ambiguous, ask clarifying follow-ups.
     DO NOT proceed with a bad query — a clear question gets better results.
  3. Produce 1-3 reformulated queries that would work well for dense
     semantic search. Remove filler words, add domain-specific terms.
  4. Identify the domain/topic so later stages can contextualize results.
  5. Output a concise analysis of what you're looking for.

OUTPUT FORMAT — write each to session state:
  refined_queries: List[str] — 1-3 optimized search queries
  original_query: str        — The user's original question
  analysis: str              — Brief: what domain, what to look for

EXAMPLE:
  User: "How does the agent runtime work?"
  refined_queries: ["Dintta agent runtime loop implementation", 
                    "agent loop propose commit escalate audit",
                    "how agent runtime reads proposes and commits actions"]
  analysis: "Engineering / Agent system — looking for the agent runtime tick loop,
             propose/commit/escalate flow, and audit_log integration"

Start by analyzing the user's question below.
"""

query_analyzer_agent = LlmAgent(
    name="QueryAnalyzer",
    model="gemini-2.5-flash",
    description=(
        "Analyzes user questions and reformulates them into "
        "optimized search queries for document retrieval."
    ),
    instruction=QUERY_ANALYZER_INSTRUCTION,
    tools=[],
    output_key="analysis",
)
