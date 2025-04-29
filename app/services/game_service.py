import time
import httpx
from sqlalchemy import and_
from sqlalchemy.orm import Session
from ..models.database import Game, StreamDetail
from ..models.schemas import GameModel
from ..utils.loggers import logger
from typing import List

class GameService:
    @staticmethod
    async def fetch_stream_details(
        client: httpx.AsyncClient,
        source_name: str,
        source_id: str,
        game_id: str,
        db: Session
    ) -> List[StreamDetail]:
        try:
            # Check if we have recent cached data (e.g., within last 5 minutes)
            current_time = int(time.time())
            cache_validity = 300  # 5 minutes in seconds
            
            cached_streams = db.query(StreamDetail).filter(
                and_(
                    StreamDetail.game_id == game_id,
                    StreamDetail.source_name == source_name,
                    StreamDetail.source_id == source_id,
                    StreamDetail.last_updated > current_time - cache_validity
                )
            ).all()
            
            if cached_streams:
                logger.info(f"Returning cached stream details for game {game_id}")
                return cached_streams
            
            # Fetch new data if cache is expired or doesn't exist
            url = f"https://streamed.su/api/stream/{source_name}/{source_id}"
            response = await client.get(url)
            response.raise_for_status()
            streams_data = response.json()
            
            # Delete old stream details
            db.query(StreamDetail).filter(
                and_(
                    StreamDetail.game_id == game_id,
                    StreamDetail.source_name == source_name,
                    StreamDetail.source_id == source_id
                )
            ).delete()
            
            # Create new stream details
            new_streams = []
            for stream_data in streams_data:
                stream_detail = StreamDetail(
                    id=f"{game_id}_{source_name}_{source_id}_{stream_data.get('streamNo')}",
                    game_id=game_id,
                    source_id=source_id,
                    source_name=source_name,
                    stream_no=stream_data.get('streamNo'),
                    language=stream_data.get('language'),
                    hd=stream_data.get('hd'),
                    embed_url=stream_data.get('embedUrl'),
                    last_updated=current_time
                )
                new_streams.append(stream_detail)
            
            db.add_all(new_streams)
            db.commit()
            
            return new_streams
            
        except Exception as e:
            logger.error(f"Error fetching stream details for {source_name}/{source_id}: {str(e)}")
            return []

    @staticmethod
    async def upsert_games(db: Session, games: List[GameModel], category: str) -> bool:
        try:
            # Skip processing for 'all' category
            if category == 'all':
                logger.info("Skipping 'all' category as per requirement")
                return True
                
            existing_games = db.query(Game).filter(Game.category == category).all()
            existing_game_ids = {game.id: game for game in existing_games}
            
            current_time = int(time.time())
            
            async with httpx.AsyncClient() as client:
                for game in games:
                    try:
                        if game.id in existing_game_ids:
                            db_game = existing_game_ids[game.id]
                            # Update existing game...
                        else:
                            db_game = Game(
                                id=game.id,
                                title=game.title,
                                category=category,
                                date=game.date,
                                popular=game.popular,
                                sources=[source.dict() for source in game.sources] if game.sources else [],
                                poster=game.poster,
                                teams=game.teams.dict() if game.teams else {}
                            )
                            db.add(db_game)
                        
                        # Fetch stream details for each source
                        if game.sources:
                            for source in game.sources:
                                if source.source and source.id:
                                    await GameService.fetch_stream_details(
                                        client,
                                        source.source,
                                        source.id,
                                        game.id,
                                        db
                                    )
                        
                    except Exception as e:
                        logger.error(f"Error processing game {game.id}: {str(e)}")
                        continue
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error in upsert_games for category {category}: {str(e)}")
            db.rollback()
            return False
