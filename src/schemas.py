from typing import List, Optional, Union, Literal, Any
from pydantic import BaseModel

class QuranMetadata(BaseModel):
    chapter: Union[str, int]
    verse: Union[str, int]
    model_config = {"extra": "allow"} 

class HadithMetadata(BaseModel):
    book: str
    hadith_number: Union[str, int]
    hadith_id: Optional[Union[str, int]]
    model_config = {"extra": "allow"}

class SearchResultItem(BaseModel):
    text: str
    metadata: Union[QuranMetadata, HadithMetadata]
    score: Optional[float]
    is_relevant: Optional[bool]
    observation: Optional[str]

class AppSearchResponse(BaseModel):
    user_question: str 
    generated_queries: Optional[List[dict]]
    results: List[SearchResultItem]
