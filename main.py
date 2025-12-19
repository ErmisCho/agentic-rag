"""
CLI entrypoint for the Agentic RAG graph.

Examples:
  python -m main --question "How do I make pizza?"
  python -m main --question "What is agent memory?" --retry-count 2 --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from graph.graph import app


LOG = logging.getLogger("agentic_rag")


@dataclass(frozen=True)
class RunConfig:
    question: str
    retry_count: int
    as_json: bool
    verbose: bool
    dotenv: bool


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(
        description="Run the Agentic RAG LangGraph app.")
    p.add_argument(
        "-q",
        "--question",
        required=True,
        help="User question to answer.",
    )
    p.add_argument(
        "--retry-count",
        type=int,
        default=0,
        help="How many times the graph is allowed to retry (0 disables retries).",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print the full result as JSON.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    p.add_argument(
        "--no-dotenv",
        dest="dotenv",
        action="store_false",
        help="Do not load environment variables from .env.",
    )
    args = p.parse_args(argv)

    if args.retry_count < 0:
        p.error("--retry-count must be >= 0")

    return RunConfig(
        question=args.question.strip(),
        retry_count=args.retry_count,
        as_json=args.as_json,
        verbose=args.verbose,
        dotenv=args.dotenv,
    )


def validate_env() -> None:
    llm_provider = os.getenv("LLM_PROVIDER")

    if not llm_provider:
        raise RuntimeError(
            "Missing required environment variable: LLM_PROVIDER "
            "(expected 'gemini' or 'ollama')."
        )

    llm_provider = llm_provider.lower()

    if llm_provider == "gemini":
        if not os.getenv("GEMINI_MODEL"):
            raise RuntimeError(
                "LLM_PROVIDER='gemini' requires GEMINI_MODEL to be set."
            )

    elif llm_provider == "ollama":
        if not os.getenv("OLLAMA_MODEL"):
            raise RuntimeError(
                "LLM_PROVIDER='ollama' requires OLLAMA_MODEL to be set."
            )

    else:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER '{llm_provider}'. "
            "Supported values are: 'gemini', 'ollama'."
        )


def run_once(cfg: RunConfig) -> Dict[str, Any]:
    payload = {"question": cfg.question, "retry_count": cfg.retry_count}
    return app.invoke(payload)


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    setup_logging(cfg.verbose)

    if cfg.dotenv:
        load_dotenv()

    try:
        validate_env()
        LOG.info("Running Agentic RAG")
        LOG.debug("Question: %s", cfg.question)

        t0 = time.perf_counter()
        result = run_once(cfg)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if cfg.as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            answer = result.get("generation")
            if not answer:
                raise RuntimeError("No 'generation' field found in result.")
            print(answer)

        LOG.info("Done in %.1f ms", dt_ms)
        return 0

    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        return 130
    except Exception:
        LOG.exception("Run failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
