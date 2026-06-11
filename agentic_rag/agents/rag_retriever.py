"""
RAG Retriever Agent — Step 2 of the Agentic RAG pipeline.

PURPOSE:
  Takes the reformulated queries from the Query Analyzer and executes
  searches against the document store using the rag_search tool.

AGENT NOTES:
  Uses gemini-2.0-flash (lightweight) since this agent primarily calls a
  tool rather than doing heavy reasoning. The tool does the actual work.
"""

try:
    from ..config import INDEX_TYPE, TOP_K
    from ..tools.rag_tool import rag_search
except ImportError:
    from config import INDEX_TYPE, TOP_K
    from tools.rag_tool import rag_search

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

_rag_tool = FunctionTool(
    func=rag_search,
    name="rag_search",
    description="Search the local document store for relevant passages using the RAG pipeline",
)


def _make_rag_retriever_agent() -> LlmAgent:
    return LlmAgent(
        name="RAGRetriever",
        model="gemini-2.0-flash",
        instruction="""You are the RAG Retrieval specialist.

Your job:
1. Read the conversation history to find the reformulated search queries from the Query Analyzer.
2. Search the RAG document store using `rag_search(query)` for each reformulated query. Try 1-2 search queries to get comprehensive coverage.
3. After searching, present a clear summary of the results you found.

Important:
- Search at least once for the main question topic.
- Report back what documents and scores you found.
- If results seem irrelevant to the original question, say so.
""",
        tools=[_rag_tool],
        output_key="rag_results",
    )


rag_retriever_agent = _make_rag_retriever_agent()
