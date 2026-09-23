"""
WaterTwin-X - optional LLM narration layer.

IMPORTANT SCOPING NOTE: this file does NOT predict water quality, does NOT
choose sampling locations, and does NOT touch the Gaussian Process engine in
any way. Its only job is turning already-computed, already-real numbers
(from real_data_report.md or a live demo alert) into a plain-English
sentence a non-technical judge or teammate can read without translation.
If this script were deleted entirely, WaterTwin-X's actual mechanism would
be completely unaffected - that's a deliberate design boundary, not an
accident. See README.md ("Should we use an LLM API?") for the full reasoning.

Uses Python's `requests` (already required elsewhere) to call the Anthropic
Messages API directly - no extra SDK install needed.

Setup:
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
    python3 llm_narrate.py                    -> narrates real_data_report.md
    python3 llm_narrate.py "your question"     -> answers a question about it
"""

import os
import sys
import json
import requests

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


def _call_claude(system_prompt, user_message, max_tokens=600):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[llm_narrate] No ANTHROPIC_API_KEY set in the environment.")
        print("[llm_narrate] Set it with:  export ANTHROPIC_API_KEY=\"sk-ant-...\"")
        print("[llm_narrate] Skipping narration - this is optional, the actual")
        print("[llm_narrate] engine and validation results are unaffected.")
        return None

    try:
        resp = requests.post(
            API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", []))
    except requests.exceptions.RequestException as e:
        print(f"[llm_narrate] API call failed: {e}")
        return None


def narrate_report(report_text):
    system_prompt = (
        "You explain water-monitoring validation results to a non-technical audience "
        "(judges, faculty, teammates) who have not seen the raw numbers before. Use plain "
        "language, short sentences, no jargon. State the honest, narrow conclusion - do not "
        "oversell the results. If a result was negative or mixed, say so plainly, don't "
        "hide it. Write 4-6 sentences, no headers, no bullet points - just a spoken-style "
        "summary someone could read aloud in a pitch."
    )
    return _call_claude(system_prompt, f"Summarize this validation report:\n\n{report_text}")


def answer_question(question, report_text):
    system_prompt = (
        "You answer questions about a water-monitoring validation report, using ONLY the "
        "numbers and findings actually in the report below - never invent a number that "
        "isn't there. If the report doesn't contain the answer, say so plainly rather than "
        "guessing. Keep answers short and in plain language."
    )
    return _call_claude(system_prompt, f"Report:\n\n{report_text}\n\nQuestion: {question}")


if __name__ == "__main__":
    try:
        with open("real_data_report.md") as f:
            report = f.read()
    except FileNotFoundError:
        print("real_data_report.md not found - run real_data_validation.py first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = answer_question(question, report)
        if answer:
            print(answer)
    else:
        summary = narrate_report(report)
        if summary:
            print(summary)
