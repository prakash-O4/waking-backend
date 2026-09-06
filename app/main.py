from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Iterator, Optional

import psycopg2

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.authority.writer import connect
from app.retrieval.gated_orchestrator import (
    answer as orchestrator_answer,
    stream_answer,
)
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


def _authorize(authorization: Optional[str]) -> str:
    supabase_helper = SupabaseHelper()
    user_id = supabase_helper.get_user_id(authorization)
    if supabase_helper.check_daily_quota(user_id):
        raise HTTPException(
            status_code=404,
            detail={"message": "Daily quota reached."},
        )
    return user_id


@app.post("/ask")
async def ask_question(
    req: AskRequest, authorization: Optional[str] = Header(default=None)
) -> Any:
    user_id = _authorize(authorization)
    as_of = req.as_of or date.today()
    try:
        with connect() as conn:
            return orchestrator_answer(req.question, as_of, conn, user_id=user_id)
    except psycopg2.OperationalError:
        return JSONResponse(
            status_code=503,
            content={
                "message": "Authority store unavailable. No validated answers can be provided.",
                "retry_after": 60,
            },
        )


@app.post("/ask/stream")
async def ask_question_stream(
    req: AskRequest, authorization: Optional[str] = Header(default=None)
) -> StreamingResponse:
    user_id = _authorize(authorization)
    as_of = req.as_of or date.today()

    def event_gen() -> Iterator[str]:
        try:
            with connect() as conn:
                for event in stream_answer(req.question, as_of, conn, user_id=user_id):
                    yield f"data: {json.dumps(event)}\n\n"
        except psycopg2.OperationalError:
            yield (
                "data: "
                + json.dumps(
                    {
                        "stage": "error",
                        "status": "error",
                        "detail": "database unavailable",
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/")
def read_root() -> dict[str, str]:
    logger.info("Checking in home")
    return {"message": "Welcome to Wakil-G!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
