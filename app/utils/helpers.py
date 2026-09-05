from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import HTTPException
from supabase import Client, create_client

from app.utils.loggers import logger

load_dotenv()


def _extract_bearer(token: str | None) -> str | None:
    if not token:
        return None
    return token.split(" ", 1)[1] if token.lower().startswith("bearer ") else token


class SupabaseHelper:
    def __init__(self) -> None:
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET")

        if not all([self.supabase_url, self.supabase_key, self.jwt_secret]):
            raise ValueError("Missing required environment variables for Supabase")

        assert self.supabase_url is not None
        assert self.supabase_key is not None
        self.project_ref = self.supabase_url.split("//")[1].split(".")[0]
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def _get_claims(self, token: str | None) -> Any | None:
        raw = _extract_bearer(token)
        if not raw:
            return None
        try:
            return self.supabase.auth.get_claims(jwt=raw)
        except Exception as e:
            logger.exception("Supabase token validation failed: {}", e)
            return None

    def validate_token(self, token: str | None) -> bool:
        """
        Validate JWT token
        """
        return self._get_claims(token) is not None

    def get_user_id(self, token: str | None) -> str:
        """
        Fetch user details from Supabase
        """
        claims = self._get_claims(token)
        if claims is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = claims["claims"].get("sub")
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")
        return str(user_id)

    def check_daily_quota(self, user_id: str) -> bool:
        """
        Check if user has exceeded the daily quota of 20 answers.
        """
        try:
            # Fetch the user's config
            response = (
                self.supabase.table("config")
                .select("*")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )

            if response.data:
                # Extract daily_limit and last_updated_at
                daily_limit = response.data[0]["daily_limit"]
                last_updated_at = response.data[0]["last_updated_at"]

                # Parse last_updated_at as a datetime object
                last_updated_date = datetime.strptime(
                    last_updated_at, "%Y-%m-%dT%H:%M:%S.%f%z"
                )

                # Get the current date in UTC
                current_date = datetime.now(timezone.utc)

                # Check if last_updated_at is not today
                if last_updated_date.date() != current_date.date():
                    # Reset daily_limit if the day has changed
                    self.supabase.table("config").update(
                        {"daily_limit": 0, "last_updated_at": current_date.isoformat()}
                    ).eq("user_id", user_id).execute()
                    daily_limit = 0  # Reset for further checks

                # Check if daily_limit exceeds or equals 20
                if daily_limit >= 20:
                    return True  # User has exceeded their daily quota
                else:
                    return False  # User can proceed
            else:
                return False

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error checking daily quota: {str(e)}"
            ) from e

    def transform_qa_data(
        self, json_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Parse JSON if input is a string
        data = json_data

        # Sort data by sent_at to maintain chronological order
        sorted_data = sorted(data, key=lambda x: x["sent_at"])

        # Initialize variables
        result = []
        current_question = None
        current_answer = None
        current_id = None

        for item in sorted_data:
            if item["sender_name"] != "bot":
                # If we have a previous Q&A pair, add it to results
                if current_question is not None and current_answer is not None:
                    result.append(
                        {
                            "id": current_id,
                            "question": current_question,
                            "answer": current_answer,
                        }
                    )
                # Start new Q&A pair
                current_question = item["content"]
                current_answer = None
                current_id = item.get("question_id")
            else:
                # This is an answer
                if current_question is not None:
                    current_answer = item["content"]

        # Add the last Q&A pair if it exists
        if current_question is not None and current_answer is not None:
            result.append(
                {
                    "id": current_id,
                    "question": current_question,
                    "answer": current_answer,
                }
            )

        return result

    def get_user_chat_history(
        self, user_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Fetch user's chat history from Supabase
        """
        #   .neq('source', None) \
        try:
            response = (
                self.supabase.table("chat")
                .select("*")
                .eq("sender_id", user_id)
                .order("sent_at", desc=True)
                .limit(limit)
                .execute()
            )
            # Extract the 'content' from each entry in response.data
            content_list = self.transform_qa_data(response.data)

            return content_list

        except Exception as e:
            raise HTTPException(
                status_code=404, detail=f"Error fetching chat history: {str(e)}"
            ) from e
