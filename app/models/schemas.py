from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    source: Optional[str] = None
    id: Optional[str] = None

class Away(BaseModel):
    name: Optional[str] = None
    badge: Optional[str] = None

class Teams(BaseModel):
    home: Optional[Away] = None
    away: Optional[Away] = None

class GameModel(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    date: Optional[int] = None
    popular: Optional[bool] = None
    sources: Optional[List[Source]] = None
    poster: Optional[str] = None
    teams: Optional[Teams] = None

    class Config:
        from_attributes = True

class StreamSource(BaseModel):
    id: Optional[str] = None
    streamNo: Optional[int] = None
    language: Optional[str] = None
    hd: Optional[bool] = None
    embedUrl: Optional[str] = None
    source: Optional[str] = None