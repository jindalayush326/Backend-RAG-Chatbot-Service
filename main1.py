from fastapi import FastAPI, UploadFile, File, HTTPException
from document_processing1 import DocumentProcessingService
from rag_chatbot1 import RAGChatbotService
from models1 import DocumentProcessResponse, ChatStartRequest, ChatStartResponse, ChatMessageRequest, ChatMessageResponse

app = FastAPI()
document_service = DocumentProcessingService()
chatbot_service = RAGChatbotService()

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Chatbot Service!"}

# Process Document Endpoint
@app.post("/api/documents/process", response_model=DocumentProcessResponse)
async def process_document(file: UploadFile = File(...)):
    try:
        # Call the document processing service with the uploaded file
        return await document_service.process_document(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start Chat Endpoint
@app.post("/api/chat/start", response_model=ChatStartResponse)
async def start_chat(request: ChatStartRequest):
    try:
        return chatbot_service.start_chat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Send Message Endpoint
@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest):
    try:
        return chatbot_service.send_message(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
