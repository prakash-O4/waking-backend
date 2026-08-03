"""
Hybrid PII redaction for NKP cases (design §4, Option C).

Stage 1 — deterministic rule pass (exact party strings, then tokens ≥4 chars).
Stage 2 — haiku second pass for residual name variants (only when stage 1
           found something; skipped safely when the SDK/key is unavailable).
Stage 3 — verification assertion: no raw party token ≥4 chars may survive;
           failure raises RedactionVerificationError and the pipeline
           quarantines the document (loud refusal, never a silent pass).
"""

from __future__ import annotations

import json
import re
import unicodedata

from app.utils.loggers import logger

APPELLANT_PLACEHOLDER = "[[वादी]]"
RESPONDENT_PLACEHOLDER = "[[प्रतिवादी]]"
MIN_TOKEN_CHARS = 4
HAIKU_MODEL = "claude-haiku-4-5-20251001"

_TOKEN_SPLIT_RE = re.compile(r"[\s,।:;()\[\]\"'\-–—/]+")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class RedactionVerificationError(Exception):
    """A raw party token survived redaction; quarantine the document."""

    def __init__(self, token: str, document_id: str):
        self.token = token
        self.document_id = document_id
        super().__init__(
            f"redaction verification failed: token {token!r} survives "
            f"in document {document_id!r}"
        )


def _digit_fold(text: str) -> str:
    """Devanagari digits → ASCII, for comparison only (stored text keeps
    the original digits)."""
    return text.translate(_DEVANAGARI_DIGITS)


def _party_variants(party: str) -> list[str]:
    nfc = unicodedata.normalize("NFC", party).strip()
    variants = {nfc, _digit_fold(nfc)}
    return sorted((v for v in variants if v), key=len, reverse=True)


def _tokens(variants: list[str]) -> set[str]:
    tokens: set[str] = set()
    for variant in variants:
        for token in _TOKEN_SPLIT_RE.split(variant):
            token = token.strip()
            # Never redact a token that is part of a placeholder itself
            # (e.g. वादी), or redaction would rewrite its own output.
            if (
                len(token) >= MIN_TOKEN_CHARS
                and token not in APPELLANT_PLACEHOLDER
                and token not in RESPONDENT_PLACEHOLDER
            ):
                tokens.add(token)
    return tokens


class PIIRedactor:
    def __init__(self, enable_llm: bool = True, llm_timeout: float = 30.0):
        self._enable_llm = enable_llm
        self._llm_timeout = llm_timeout

    def redact(
        self,
        full_text: str,
        appellant: str,
        respondent: str,
        document_id: str = "",
    ) -> tuple[str, list[str]]:
        """
        Returns: (redacted_text, list_of_redaction_warnings)
        Raises: RedactionVerificationError if any raw token ≥4 chars survives.
        """
        warnings: list[str] = []
        text = unicodedata.normalize("NFC", full_text)

        parties = [
            (appellant or "", APPELLANT_PLACEHOLDER),
            (respondent or "", RESPONDENT_PLACEHOLDER),
        ]

        # Stage 1 — deterministic.
        replacements = 0
        all_tokens: set[str] = set()
        for party, placeholder in parties:
            variants = _party_variants(party)
            all_tokens |= _tokens(variants)
            for variant in variants:
                count = text.count(variant)
                if count:
                    text = text.replace(variant, placeholder)
                    replacements += count
        for party, placeholder in parties:
            party_tokens = _tokens(_party_variants(party))
            for token in sorted(party_tokens, key=len, reverse=True):
                count = text.count(token)
                if count:
                    text = text.replace(token, placeholder)
                    replacements += count

        # Stage 2 — haiku second pass. Only when stage 1 found PII; if stage 1
        # replaced nothing it already confirmed no verbatim PII is present.
        if replacements > 0 and self._enable_llm:
            try:
                text = self._haiku_second_pass(text, appellant, respondent)
            except Exception as exc:  # noqa: BLE001 — never crash on the LLM pass
                logger.warning(f"PII haiku second pass failed, continuing: {exc}")
                warnings.append(f"haiku_second_pass_failed: {exc}")

        # Stage 3 — verification assertion (the safety net).
        self._verify(text, all_tokens, document_id)
        return text, warnings

    def _verify(self, redacted_text: str, tokens: set[str], document_id: str) -> None:
        for token in sorted(tokens):
            if token in redacted_text:
                raise RedactionVerificationError(token=token, document_id=document_id)

    def _haiku_second_pass(self, text: str, appellant: str, respondent: str) -> str:
        import anthropic  # lazy: SDK is an ingestion-time-only dependency

        client = anthropic.Anthropic(timeout=self._llm_timeout)
        prompt = (
            f"Given these party names: [{appellant}], [{respondent}] — identify any "
            "variant mentions (abbreviated names, honorifics, partial names) in the "
            "following text and return a JSON list of spans to replace. Format: "
            '[{"original": "...", "replacement": "[[वादी]]"}]. Use [[वादी]] for the '
            "appellant and [[प्रतिवादी]] for the respondent. Return ONLY the JSON "
            f"list.\n\nText:\n{text}"
        )
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        spans = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
        for span in spans:
            original = str(span.get("original") or "")
            replacement = str(span.get("replacement") or APPELLANT_PLACEHOLDER)
            if original:
                text = text.replace(original, replacement)
        return text
