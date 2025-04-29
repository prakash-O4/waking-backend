import hmac
import hashlib
from fastapi import HTTPException, Header
from app.config import get_settings

settings = get_settings()

def verify_hmac(signature: str = Header(...), timestamp: str = Header(...), body: str = ""):
    if not signature or not timestamp:
        raise HTTPException(status_code=401, detail="Missing authentication headers")
    
    msg = f"{timestamp}.{body}".encode('utf-8')
    expected_signature = hmac.new(
        settings.API_SECRET.encode('utf-8'),
        msg,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")