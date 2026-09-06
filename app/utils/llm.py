from __future__ import annotations

from typing import Any


def llm_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text:
        return text
    if callable(text):
        try:
            value = text()
            if isinstance(value, str) and value:
                return value
        except Exception:
            pass

    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
