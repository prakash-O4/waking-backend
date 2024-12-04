from typing import  Dict, List
import jwt
from supabase import create_client, Client
from fastapi import HTTPException, Header
import os
from dotenv import load_dotenv

load_dotenv()

class SupabaseHelper:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        
        if not all([self.supabase_url, self.supabase_key, self.jwt_secret]):
            raise ValueError("Missing required environment variables for Supabase")
        
        self.project_ref = self.supabase_url.split('//')[1].split('.')[0]
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)



    def validate_token(self, token: str) -> bool:
        """
        Validate JWT token
        """
        try:
            self.supabase.auth._decode_jwt(token)
            # payload  = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], audience="authenticated")
            return True
        except jwt.ExpiredSignatureError:
            return False
        except Exception as e:
            return False
        
    def get_user_id(self, token: str) -> Dict:
        """
        Fetch user details from Supabase
        """
        try:
            if not self.validate_token(token):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token"
                )
            jwtToken = token.split(" ")[1]
            response = self.supabase.auth.get_user(jwtToken)
            
            if not response.user:
                raise HTTPException(
                    status_code=404,
                    detail="User not found"
                )
            
            return response.user.id

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching user details: {str(e)}"
            )

    def get_user_chat_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Fetch user's chat history from Supabase
        """
        try:
            response = self.supabase.table('chat') \
                          .select("*") \
                          .eq('sender_id', user_id) \
                          .neq('source', None) \
                          .order('sent_at', desc=True) \
                          .limit(limit) \
                          .execute()

            # Extract the 'content' from each entry in response.data
            content_list = [item['content'] for item in response.data] if response.data else []

            return content_list


        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching chat history: {str(e)}"
            )
