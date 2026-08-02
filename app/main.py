from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Optional, cast

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.authority.writer import connect
from app.retrieval.dumb_retriever import retrieve
from app.retrieval.validation_gate import validate_and_render
from app.utils.helpers import SupabaseHelper
from app.utils.loggers import logger

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Wakil-G"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    as_of: Optional[date] = None


def _claims_from_model(
    question: str, hits: list[dict[str, Any]]
) -> dict[str, Any] | None:
    context = "\n\n---\n\n".join(
        f"[{h['component_uri']}]\n{h['text_ne']}" for h in hits
    )
    system = (
        "You are Wakil-G. Answer using ONLY the provided context. "
        "Output JSON only — no markdown, no explanation outside the JSON:\n"
        '{"claims": [{"claim": "<answer text>", '
        '"evidence_id": "<component_uri from context>"}]}\n'
        'If context is insufficient: {"claims": [], "abstain": true}\n'
        "Do not write citations. Do not include anything not in the context."
    )
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.0,
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    )
    parsed = json.loads(response.content.strip())
    return cast(dict[str, Any], parsed)


@app.post("/ask")
async def ask_question(
    req: AskRequest, authorization: Optional[str] = Header(default=None)
) -> dict[str, Any]:
    supabase_helper = SupabaseHelper()
    user_id = supabase_helper.get_user_id(authorization)
    if supabase_helper.check_daily_quota(user_id):
        raise HTTPException(
            status_code=404,
            detail={"message": "Daily quota reached."},
        )

    as_of = req.as_of or date.today()
    hits = retrieve(req.question, as_of, k=5)
    if not hits:
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    try:
        parsed = _claims_from_model(req.question, hits)
    except Exception as exc:
        logger.error(f"Model call or parse failed: {exc}")
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    if not parsed or parsed.get("abstain") or not parsed.get("claims"):
        return {"as_of": as_of.isoformat(), "abstained": True, "results": []}

    with connect() as conn:
        validated = validate_and_render(parsed["claims"], as_of, conn)

    return {"as_of": as_of.isoformat(), "abstained": False, "results": validated}


@app.get("/")
def read_root() -> dict[str, str]:
    logger.info("Checking in home")
    return {"message": "Welcome to Wakil-G!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
