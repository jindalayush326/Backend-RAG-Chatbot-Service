from pydantic import BaseModel

class DocumentProcessRequest(BaseModel):
    file_path: str

class DocumentProcessResponse(BaseModel):
    asset_id: str

class ChatStartRequest(BaseModel):
    asset_id: str

class ChatStartResponse(BaseModel):
    chat_id: str

class ChatMessageRequest(BaseModel):
    chat_id: str
    message: str

class ChatMessageResponse(BaseModel):
    response: str
