# app/models.py
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    subject: str
    max_retries: int = 2
    thread_id: Optional[str] = None  # chat session ID
    user_id: Optional[str] = "default_user"  # for LTM user profile


class SourceItem(BaseModel):
    doc_id: str
    book_name: Optional[str] = None
    content_type: str
    pages: List[int]
    text: str
    image_path: Optional[str] = None
    formula_image_path: Optional[str] = None
    table_image_path: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    is_grounded: bool
    retry_count: int
    intent: Optional[str] = None
    sources: List[SourceItem]
    thread_id: str  # always returned so frontend can track session


class ChatSession(BaseModel):
    thread_id: str
    title: str
    subject: str
    created_at: str
    last_message: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    sessions: List[ChatSession]