from pydantic import BaseModel, Field#, Optional
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    LLAMA = "llama3-8b-8192"
   # GPT4_MINI = "gpt-4o-mini"


class QueryInput(BaseModel):
    question: str
    session_id: str | None = None
    model: ModelName

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int
