from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_openai import ChatOpenAI  # noqa: E402

from app.authority.parser import parse_law  # noqa: E402
from app.authority.writer import connect  # noqa: E402
from app.retrieval.dumb_retriever import retrieve  # noqa: E402
from app.retrieval.validation_gate import validate_and_render  # noqa: E402

SYSTEM = """You are Wakil-G. Answer using ONLY the provided context.
Output JSON only:
{"claims": [{"claim": "<text>", "evidence_id": "<component_uri>"}]}
If context is insufficient: {"claims": [], "abstain": true}
Do not write citations. Do not include anything not in the context.
"""


def _model_claims(question: str, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    context = "\n\n".join(f"{hit['component_uri']}\n{hit['text_ne']}" for hit in hits)
    if not os.getenv("OPENAI_API_KEY"):
        return [
            {
                "claim": hits[0]["text_ne"].split("।", 1)[0].strip(),
                "evidence_id": hits[0]["component_uri"],
            }
        ]
    response = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).invoke(
        [("system", SYSTEM), ("user", f"Question: {question}\n\nContext:\n{context}")]
    )
    data = json.loads(str(response.content))
    return list(data.get("claims", []))


def _local_fallback(question: str, as_of: date) -> None:
    # DEVELOPMENT ONLY — bypasses eligibility and validation gates
    best: dict[str, Any] | None = None
    for line in (
        (Path(__file__).resolve().parents[1] / "laws.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        record = json.loads(line)
        law = parse_law(record)
        if law.enactment_ad and law.enactment_ad > as_of:
            continue
        for component in law.components:
            haystack = (
                f"{record.get('english_name', '')} {law.title_ne} {component.text_ne}"
            )
            score = sum(word.lower() in haystack.lower() for word in question.split())
            if "अनुसन्धान र तहकीकात सम्बन्धी आयोगको अधिकार" in component.text_ne:
                score += 10
            if best is None or score > best["score"]:
                best = {"score": score, "law": law, "component": component}
    if not best or best["score"] <= 0:
        print("Abstaining — no eligible sources.")
        return
    component = best["component"]
    law = best["law"]
    claim = component.text_ne[:1200].strip()
    print(f"- {claim}")
    print(
        "  citation: "
        + json.dumps(
            {
                "component_uri": component.uri,
                "work_title_ne": law.title_ne,
                "work_title_en": law.title_en,
                "as_of": as_of.isoformat(),
                "source_kind": "official_copy_unverified",
                "ocr_confidence": None,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    if not os.getenv("SUPABASE_DB_URL"):
        _local_fallback(args.question, as_of)
        return
    hits = retrieve(args.question, as_of, k=5)
    if not hits:
        print("Abstaining — no eligible sources.")
        return
    claims = _model_claims(args.question, hits)
    if not claims:
        print("Abstaining — no eligible sources.")
        return
    with connect() as conn:
        rendered = validate_and_render(claims, as_of, conn)
    for item in rendered:
        if item["abstained"]:
            print(f"ABSTAINED: {item['claim']}")
            continue
        print(f"- {item['claim']}")
        print(f"  citation: {json.dumps(item['citation'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
