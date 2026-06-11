"""
Agentic RAG — CLI Runner.

PURPOSE:
  One-shot CLI for the Agentic RAG pipeline. Runs the query through
  all 5 pipeline stages with retry for API rate limits.

USAGE:
  python run.py "What is the three-plane architecture of Dintta?"
  python run.py --verbose "How does JWT auth work?"
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types

from agent import agent

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)


def run_pipeline(
    question: str, verbose: bool = False, max_retries: int = 3
) -> str:
    """Run the pipeline with retry for API rate limits."""
    app_name = "agentic_rag_cli"
    user_id = "user"
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=question)],
    )
    last_error = None

    for attempt in range(1 + max_retries):
        session_id = f"session_{int(time.time() * 1000)}_{attempt}"
        final_response = ""

        if attempt > 0:
            wait = 60  # 60s cooldown for Vertex AI rate limits
            print(
                f"  Retry {attempt}/{max_retries} after {wait}s..."
            )
            time.sleep(wait)

        try:
            async def _run():
                nonlocal final_response
                svc = InMemorySessionService()
                await svc.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                runner = Runner(
                    agent=agent,
                    app_name=app_name,
                    session_service=svc,
                )
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=user_content,
                ):
                    if verbose:
                        _print_event(event)
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            parts_text = [
                                p.text for p in event.content.parts if p.text
                            ]
                            if parts_text:
                                final_response = "".join(parts_text)

            asyncio.run(_run())
            if final_response:
                return final_response

        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            raise

    raise last_error  # type: ignore[misc]


def _print_event(event):
    """Print intermediate events for verbose mode."""
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.text:
            print(f"\n-- [LLM] --")
            print(part.text[:500])
            if len(part.text) > 500:
                print("  ...(truncated)")
        elif part.function_call:
            fc = part.function_call
            print(f"\n-- [LLM -> tool: {fc.name}] --")
            print(f"    args: {dict(fc.args) if fc.args else {}}")
        elif part.function_response:
            fr = part.function_response
            brief = str(fr.response)[:300]
            print(f"\n-- [tool: {fr.name} -> LLM] --")
            print(f"    result: {brief}")


def main():
    """Parse args and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG - multi-stage document + web QA"
    )
    parser.add_argument("question", nargs="*", help="Your question")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all stages"
    )
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        return

    question = " ".join(args.question)
    print(f"\nAgentic RAG Pipeline (single agent)")
    print(f"  Question: {question}")
    print(f"  Model: gemini-2.5-flash")
    print(f"  Tools: rag_search + web_search")
    print(f"  Stages: internal 5-stage pipeline (1 API call)\n")

    try:
        answer = run_pipeline(question, verbose=args.verbose)
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER\n")
        print(answer)
        print(f"\n{'='*60}")
    except Exception as e:
        print(f"\nError after retries: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
