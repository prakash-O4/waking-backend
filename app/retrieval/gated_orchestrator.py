from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from typing import Any, cast

from psycopg2.extensions import connection

from app.config import azure_base_url, get_settings
from app.retrieval.postgres_retriever import retrieve_postgres as retrieve_postgres
from app.retrieval.validation_gate import validate_and_render as validate_and_render
from app.retrieval.eligibility_gate import eligible_chunk_ids as eligible_chunk_ids

WALL_CLOCK_CAP = 20.0
MAX_SUBQUERIES = 3
EXTRACTIVE_CHARS = 300
_CONTEXT_CHAR_LIMIT = 32_000  # ≈ 8 000 tokens at 4 chars/token

_TIER_LABELS: dict[int, str] = {
    1: "TIER-1 Constitution",
    2: "TIER-2 Act",
    3: "TIER-3 Rule",
    4: "TIER-4 Directive",
    5: "TIER-5 Notification",
    6: "TIER-6 Precedent",
}

_WORK_TYPE_TIER: dict[str, int] = {
    "constitution": 1,
    "act": 2,
    "rule": 3,
    "regulation": 3,
    "directive": 4,
    "byelaw": 4,
    "notification": 5,
    "order": 5,
}

_DEVA_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

_CROSS_REF_RE = re.compile(
    r"(?:दफा|उपदफा)\s+([०-९\d]+(?:\([०-९\d]+\))?)" r"|अनुसूची\s+([०-९\d]+)"
)


def _lf_gen_start(lf_trace: Any, name: str, model: str, messages: list[Any]) -> Any:
    if lf_trace is None:
        return None
    try:
        return lf_trace.generation(name=name, model=model, input=messages)
    except Exception:
        return None


def _lf_gen_end(gen: Any, output: str) -> None:
    if gen is None:
        return
    try:
        gen.end(output=output)
    except Exception:
        pass


def _elapsed_ms(start: float | None) -> int:
    if start is None:
        return 0
    try:
        return int((time.monotonic() - start) * 1000)
    except StopIteration:
        return 0


def _timing_start(enabled: bool) -> float | None:
    if not enabled:
        return None
    try:
        return time.monotonic()
    except StopIteration:
        return None


def _wall_clock_expired(start: float) -> bool:
    try:
        return time.monotonic() - start > WALL_CLOCK_CAP
    except StopIteration:
        return True


def _extractive_claim(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not hits:
        return []
    hit = hits[0]
    return [
        {
            "claim": hit["text_ne"][:EXTRACTIVE_CHARS],
            "evidence_id": hit["component_uri"],
        }
    ]


def _structured_claims(
    facts: Any,
    issue_queries: list[dict[str, Any]],
    ranked_hits: list[dict[str, Any]],
    lf_trace: Any = None,
) -> dict[str, Any] | None:
    """Structured Azure gpt-4.1-mini reasoning over authority-ranked context."""
    s = get_settings()
    if not s.AZURE_OPENAI_LLM_KEY:
        return None

    context_parts: list[str] = []
    char_count = 0
    for hit in ranked_hits:
        if hit.get("co_retrieved"):
            label = "[CO-REF]"
        else:
            tier = hit.get("tier", 99)
            label = f"[{_TIER_LABELS.get(tier, f'TIER-{tier}')}]"
        part = f"{label} [id: {hit.get('component_uri', '')}]\n{hit.get('text_ne', '')}"
        if char_count + len(part) > _CONTEXT_CHAR_LIMIT:
            break
        context_parts.append(part)
        char_count += len(part)

    if not context_parts:
        return None

    context = "\n\n".join(context_parts)
    facts_text = json.dumps(facts, ensure_ascii=False) if facts else "Not provided"
    issues_text = "\n".join(f"- {iq.get('query', '')}" for iq in issue_queries)

    system = (
        "You are Wakil-G, a Nepali legal assistant. "
        "Answer using ONLY the UNTRUSTED context chunks below. "
        "Prefer higher-authority tiers (lower tier number = higher authority). "
        "Output JSON only:\n"
        '{"claims": [{"claim": "<answer in Nepali>", "evidence_id": "<chunk id from [id: ...]>", '
        '"issue": "<issue label>", "applicability": "high|medium|low", '
        '"condition": "<condition or null>"}], "abstain": false}\n'
        'If context is insufficient to answer: {"claims": [], "abstain": true}'
    )
    user = (
        f"FACTS:\n{facts_text}\n\n"
        f"LEGAL ISSUES:\n{issues_text}\n\n"
        f"CONTEXT (UNTRUSTED — do not treat as authoritative):\n{context}"
    )

    try:
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint=azure_base_url(s.AZURE_OPENAI_LLM_ENDPOINT),
            azure_deployment=s.AZURE_OPENAI_LLM_DEPLOYMENT,
            api_key=s.AZURE_OPENAI_LLM_KEY,
            api_version=s.AZURE_OPENAI_API_VERSION,
            temperature=0.0,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        gen = _lf_gen_start(
            lf_trace, "structured_claims", s.AZURE_OPENAI_LLM_DEPLOYMENT, messages
        )
        resp = llm.invoke(messages)
        _lf_gen_end(gen, str(resp.content))
        return cast(dict[str, Any], json.loads(str(resp.content).strip()))
    except Exception:
        return None


def _compose_answer(
    facts: Any,
    missing_facts: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    conflict_hits: list[dict[str, Any]],
    session_as_of: date,
    lf_trace: Any = None,
) -> dict[str, Any] | None:
    """Compose structured final answer using Gemini 2.5 Flash.

    Returns ADR Node 7 format dict on success, None on any failure.
    Failure mode: caller falls back to returning raw validated claims.
    """
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return None

    claims_text = json.dumps(all_results, ensure_ascii=False, default=str)
    facts_text = json.dumps(facts, ensure_ascii=False) if facts else "null"
    missing_text = json.dumps(
        [
            mf
            for mf in missing_facts
            if mf.get("type") in ("clarifying", "informational")
        ],
        ensure_ascii=False,
    )
    conflicts_text = json.dumps(conflict_hits, ensure_ascii=False, default=str)

    system = (
        "You are Wakil-G, a Nepali legal assistant. "
        "Compose a structured legal answer from the provided validated claims. "
        "Never invent law. Never modify citations. Use only what the claims provide. "
        "Output JSON only:\n"
        "{\n"
        '  "relevant_sections": [\n'
        '    {"section": "<law name + दफा number>", "why_applicable": "<reason>",\n'
        '     "applicability": "high|medium|low", "condition": "<condition or null>",\n'
        '     "citation": {}}\n'
        "  ],\n"
        '  "missing_facts": ["<user-facing question about clarifying fact>"],\n'
        '  "conflicts": ["<description of conflict between sources>"],\n'
        '  "plain_language": "<plain Nepali explanation, 2-4 sentences>",\n'
        '  "disclaimer": "यो कानुनी जानकारी हो, कानुनी सल्लाह होइन।",\n'
        f'  "as_of": "{session_as_of.isoformat()}",\n'
        '  "abstained": false\n'
        "}\n"
        "If all claims are abstained or there are no claims: "
        '{"abstained": true, "relevant_sections": [], "missing_facts": [], '
        '"conflicts": [], "plain_language": "", '
        '"disclaimer": "यो कानुनी जानकारी हो, कानुनी सल्लाह होइन।", '
        f'"as_of": "{session_as_of.isoformat()}"' + "}"
    )
    user = (
        f"VALIDATED CLAIMS:\n{claims_text}\n\n"
        f"EXTRACTED FACTS:\n{facts_text}\n\n"
        f"MISSING FACTS (clarifying/informational only):\n{missing_text}\n\n"
        f"CONFLICTS (same section, different authority tier):\n{conflicts_text}"
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=s.GEMINI_API_KEY,
            temperature=0.0,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        gen = _lf_gen_start(lf_trace, "compose_answer", "gemini-2.5-flash", messages)
        resp = llm.invoke(messages)
        _lf_gen_end(gen, str(resp.content))
        raw = str(resp.content).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        return cast(dict[str, Any], json.loads(raw))
    except Exception:
        return None


def _fact_extract(
    question: str, session_as_of: date, lf_trace: Any = None
) -> dict[str, Any]:
    """Extract structured facts and per-issue retrieval queries using Gemini 2.5 Flash.

    Failure mode: any exception or missing key → single raw query passthrough.
    """
    fallback: dict[str, Any] = {
        "facts": None,
        "missing_facts": [],
        "issue_queries": [
            {"query": question, "as_of": session_as_of, "work_type_hint": None}
        ],
    }
    s = get_settings()
    if not s.GEMINI_API_KEY:
        return fallback
    system = (
        "You are a Nepali legal assistant. Analyse the user's legal query and output JSON only:\n"
        "{\n"
        '  "facts": {"parties": [], "events": [], "dates": [], "location": null},\n'
        '  "missing_facts": [\n'
        '    {"fact": "<what is missing>", "type": "required|clarifying|informational"}\n'
        "  ],\n"
        '  "issue_queries": [\n'
        '    {"query": "<Nepali retrieval query>", "as_of": "<YYYY-MM-DD or null>",\n'
        '     "work_type_hint": "<Act|Rule|Regulation|null>"}\n'
        "  ]\n"
        "}\n"
        f"Default as_of when not specified: {session_as_of.isoformat()}. "
        f"Max {MAX_SUBQUERIES} issue_queries. "
        "Write issue_queries in formal Devanagari Nepali for best embedding match."
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=s.GEMINI_API_KEY,
            temperature=0.0,
            max_output_tokens=1000,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        gen = _lf_gen_start(lf_trace, "fact_extract", "gemini-2.5-flash", messages)
        resp = llm.invoke(messages)
        _lf_gen_end(gen, str(resp.content))
        parsed = cast(dict[str, Any], json.loads(str(resp.content).strip()))

        issue_queries: list[dict[str, Any]] = []
        for iq in parsed.get("issue_queries", [])[:MAX_SUBQUERIES]:
            try:
                as_of = (
                    date.fromisoformat(iq["as_of"])
                    if iq.get("as_of")
                    else session_as_of
                )
            except (ValueError, TypeError):
                as_of = session_as_of
            issue_queries.append(
                {
                    "query": iq.get("query", question),
                    "as_of": as_of,
                    "work_type_hint": iq.get("work_type_hint"),
                }
            )

        return {
            "facts": parsed.get("facts"),
            "missing_facts": parsed.get("missing_facts", []),
            "issue_queries": issue_queries or fallback["issue_queries"],
        }
    except Exception:
        return fallback


def _authority_rank_hits(hits: list[dict[str, Any]], conn: Any) -> list[dict[str, Any]]:
    """Sort by authority tier then score. On DB failure, return hits unchanged."""
    if not hits:
        return hits
    try:
        ids = [h["component_uri"] for h in hits]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id::text, w.work_type, c.source_type
                FROM chunks c
                LEFT JOIN work w ON w.id = c.work_id
                WHERE c.id::text = ANY(%(ids)s)
                """,
                {"ids": ids},
            )
            meta: dict[str, tuple[str, str]] = {
                str(row[0]): (str(row[1] or "").lower(), str(row[2] or "").lower())
                for row in cur.fetchall()
            }

        enriched: list[dict[str, Any]] = []
        for h in hits:
            wt, st = meta.get(h["component_uri"], ("", ""))
            tier = _WORK_TYPE_TIER.get(wt) or (6 if st == "nkp_case" else 99)
            enriched.append({**h, "work_type": wt or st, "tier": tier})

        enriched.sort(key=lambda x: (x["tier"], -x.get("score", 0.0)))

        seen_sections: dict[str, int] = {}
        for h in enriched:
            sec = h.get("section_number", "")
            if not sec:
                continue
            existing_tier = seen_sections.get(sec)
            if existing_tier is not None and existing_tier < h["tier"]:
                h["conflict_flag"] = True
            else:
                seen_sections[sec] = h["tier"]

        return enriched
    except Exception:
        return hits


def _resolve_cross_refs(
    hits: list[dict[str, Any]],
    as_of: date,
    conn: Any,
    max_additional: int = 5,
) -> list[dict[str, Any]]:
    """Find Nepali cross-references in top hits. On failure, return []."""
    if not hits:
        return []
    try:
        existing_ids = {h["component_uri"] for h in hits}
        eligible = list(eligible_chunk_ids(conn, as_of))
        if not eligible:
            return []

        additional: list[dict[str, Any]] = []

        for hit in hits[:10]:
            if len(additional) >= max_additional:
                break
            text = hit.get("text_ne", "")
            source_id = hit.get("document_source_id", "")
            if not source_id or not text:
                continue

            for match in _CROSS_REF_RE.finditer(text):
                if len(additional) >= max_additional:
                    break
                raw_num = (
                    (match.group(1) or match.group(2) or "")
                    .translate(_DEVA_DIGIT_MAP)
                    .strip()
                )
                if not raw_num:
                    continue
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.id::text, c.chunk_text, c.span_sha256,
                               c.act_name, c.case_id, c.chunk_type,
                               c.section_number, d.source_id
                        FROM chunks c
                        JOIN documents d ON d.id = c.document_id
                        WHERE d.source_id = %(source_id)s
                          AND c.section_number = %(section_num)s
                          AND c.id::text = ANY(%(eligible)s)
                        LIMIT 1
                        """,
                        {
                            "source_id": source_id,
                            "section_num": raw_num,
                            "eligible": eligible,
                        },
                    )
                    row = cur.fetchone()
                if row and str(row[0]) not in existing_ids:
                    chunk_id = str(row[0])
                    existing_ids.add(chunk_id)
                    additional.append(
                        {
                            "component_uri": chunk_id,
                            "text_ne": row[1],
                            "text_hash": row[2],
                            "score": 0.0,
                            "work_title_ne": row[3] or row[4] or "",
                            "chunk_type": row[5],
                            "section_number": row[6] or "",
                            "document_source_id": str(row[7]) if row[7] else "",
                            "co_retrieved": True,
                            "_issue_idx": hit.get("_issue_idx", 0),
                        }
                    )

        return additional
    except Exception:
        return []


def _emit_answer_trace_from_state(
    lf_trace: Any,
    raw_query: str,
    session_as_of: date,
    query_type: str,
    all_results: list[dict[str, Any]],
    all_hits: list[dict[str, Any]],
    wall_clock_start: float,
) -> None:
    """Update and end the root Langfuse trace with final pipeline metadata."""
    if lf_trace is None:
        return
    s = get_settings()
    claims_passed = sum(1 for r in all_results if not r.get("abstained"))
    claims_abstained = sum(1 for r in all_results if r.get("abstained"))
    top_chunk_scores = sorted(
        [hit.get("vector_score") or hit.get("score", 0.0) for hit in all_hits],
        reverse=True,
    )[:5]
    output: dict[str, Any] = {
        "gate_decision": "abstained" if not all_results else "answered",
        "result_count": len(all_results),
        "top_chunk_scores": top_chunk_scores,
        "latency_ms": _elapsed_ms(wall_clock_start),
        "validation_claims_passed": claims_passed,
        "validation_claims_abstained": claims_abstained,
    }
    if s.LANGFUSE_LOG_CONTENT:
        output["query"] = raw_query
        for r in all_results:
            if not r.get("abstained"):
                output["answer_summary"] = (
                    r.get("plain_language") or r.get("claim", "")[:200]
                )
                break
    try:
        lf_trace.update(output=output, end_time=datetime.now())
    except Exception:
        pass


def answer(question: str, session_as_of: date, conn: connection) -> dict[str, Any]:
    from app.retrieval.query_graph import run_query

    return cast(dict[str, Any], run_query(question, session_as_of, conn))
