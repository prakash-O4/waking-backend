from sqlalchemy import create_engine, Column, String, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

settings = get_settings()
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Game(Base):
    __tablename__ = "games"
    
    id = Column(String, primary_key=True)
    title = Column(String)
    category = Column(String)
    date = Column(Integer)
    popular = Column(Boolean)
    sources = Column(JSON)
    poster = Column(String)
    teams = Column(JSON)

class StreamDetail(Base):
    __tablename__ = "stream_details"
    
    id = Column(String, primary_key=True)
    game_id = Column(String, index=True)
    source_id = Column(String)
    source_name = Column(String)
    stream_no = Column(Integer)
    language = Column(String)
    hd = Column(Boolean)
    embed_url = Column(String)
    last_updated = Column(Integer)

Base.metadata.create_all(bind=engine)