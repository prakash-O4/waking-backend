from typing import List, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from sqlalchemy.orm import Session

from .models.database import SessionLocal, StreamDetail
from .models.schemas import GameModel, StreamSource
from .services.game_service import GameService
from .utils.loggers import logger
from .utils.security import verify_hmac

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

API_ENDPOINTS = [
    {"key": "live", "url": "https://streamed.su/api/matches/live/popular"},
    {"key": "football", "url": "https://streamed.su/api/matches/football/popular"},
    {"key": "basketball", "url": "https://streamed.su/api/matches/basketball/popular"},
    {"key": "american-football", "url": "https://streamed.su/api/matches/american-football/popular"},
    {"key": "hockey", "url": "https://streamed.su/api/matches/hockey/popular"},
    {"key": "fight", "url": "https://streamed.su/api/matches/fight/popular"},
    {"key": "all", "url": "https://streamed.su/api/matches/all"}
]

class GameResponseModel(GameModel):
    stream_details: Optional[List[StreamSource]] = None

@app.post("/api/fetch-games")
async def fetch_games(
    db: Session = Depends(get_db),
    # signature: str = Header(...),
    # timestamp: str = Header(...),
    category: Optional[str] = None
):
    # verify_hmac(signature=signature, timestamp=timestamp)
    
    async with httpx.AsyncClient() as client:
        endpoints_to_fetch = [ep for ep in API_ENDPOINTS if category is None or ep["key"] == category]
        
        if category and not endpoints_to_fetch:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        for endpoint in endpoints_to_fetch:
            try:
                logger.info(f"Fetching games for category: {endpoint['key']}")
                response = await client.get(endpoint["url"])
                response.raise_for_status()
                games_data = response.json()
                
                # Convert the raw data to GameModel objects
                games = [GameModel(**game_data) for game_data in games_data]
                
                # Perform upsert operation
                success = await GameService.upsert_games(db, games, endpoint["key"])
                
                if success:
                    logger.info(f"Successfully updated games for {endpoint['key']}")
                else:
                    logger.error(f"Failed to update games for {endpoint['key']}")
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching games from {endpoint['key']}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing games from {endpoint['key']}: {str(e)}")
                continue
    
    return {"message": "Games fetched and updated successfully"}

@app.get("/api/games", response_model=List[GameResponseModel])
async def get_games(
    category: Optional[str] = None,
    include_streams: bool = False,
    db: Session = Depends(get_db)
):
    try:
        games = GameService.get_games(db, category)
        
        if include_streams:
            response_games = []
            for game in games:
                # Convert SQL Alchemy model to Pydantic model
                game_dict = {
                    **game.__dict__,
                    'stream_details': []
                }
                
                # Get stream details if requested
                stream_details = db.query(StreamDetail).filter(
                    StreamDetail.game_id == game.id
                ).all()
                
                # Convert stream details to response model
                game_dict['stream_details'] = [
                    StreamSource(
                        id=str(detail.source_id),
                        streamNo=detail.stream_no,
                        language=detail.language,
                        hd=detail.hd,
                        embedUrl=detail.embed_url,
                        source=detail.source_name
                    ) for detail in stream_details
                ]
                
                response_games.append(GameResponseModel(**game_dict))
            
            return response_games
        
        return games
        
    except Exception as e:
        logger.error(f"Error retrieving games: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")