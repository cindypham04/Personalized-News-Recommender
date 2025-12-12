from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "News API alive"}

class SummarizeRequest(BaseModel):
    # request body for summarization endpoint
    query: str 
    user_id: Optional[str] = None
    reading_time: Optional[str] = None # e.g. "30s"

class ArticleSummary(BaseModel):
    # response body for summarization endpoint
    title: str
    summary: str
    url: Optional[str] = None

class SummarizeResponse(BaseModel):
    articles: List[ArticleSummary]

@app.post("/search_and_summarize", response_model=SummarizeResponse)
async def search_and_summarize(req: SummarizeRequest):
    # for now: placeholder
    return SummarizeResponse(articles=[])
