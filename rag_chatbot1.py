import uuid
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from models1 import ChatStartRequest, ChatStartResponse, ChatMessageRequest, ChatMessageResponse
from utils1 import load_vector_db
from fastapi import HTTPException

# Load the vector database and define sessions
vector_db = load_vector_db()
chat_sessions = {}

class RAGChatbotService:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize the text generation model (e.g., GPT-2 or other language model)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_generator = AutoModelForCausalLM.from_pretrained("gpt2")
        self.generator = pipeline("text-generation", model=self.text_generator, tokenizer=self.tokenizer)

    def start_chat(self, request: ChatStartRequest) -> ChatStartResponse:
        chat_id = str(uuid.uuid4())
        chat_sessions[chat_id] = {'asset_id': request.asset_id, 'history': []}
        return ChatStartResponse(chat_id=chat_id)

    def send_message(self, request: ChatMessageRequest) -> ChatMessageResponse:
        chat_data = chat_sessions.get(request.chat_id)
        if not chat_data:
            raise HTTPException(status_code=404, detail="Chat not found")

        asset_id = chat_data['asset_id']
        
        # Generate embedding for the user message
        user_embedding = self.embedding_model.encode(request.message)
        
        # Perform similarity search in the vector database
        results = vector_db.query(
            query_embeddings=[user_embedding],
            num_results=1,               # Try `num_results` or other variations if `top_k` and `limit` don't work
            include_metadatas=True
        )
        
        # Extract the document from the results to use in chat context
        documents = [doc['metadata'] for doc in results['documents']] if 'documents' in results else []
        
        # Generate a response using the language model
        user_message = request.message
        context = " ".join([doc['content'] for doc in documents]) if documents else ""
        prompt = f"{context}\n\nUser: {user_message}\nBot:"

        # Generate text response
        agent_response = self.generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        bot_response = agent_response.split("Bot:")[-1].strip()  # Extract only bot's response
        
        chat_data['history'].append({'user': user_message, 'bot': bot_response})
        
        return ChatMessageResponse(response=bot_response)
