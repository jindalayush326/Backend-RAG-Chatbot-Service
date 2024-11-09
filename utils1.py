import hashlib
from chromadb import Client
from fastapi import HTTPException


def generate_asset_id(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()

def save_embeddings(embedding, metadata, vector_db):
    asset_id = metadata['asset_id']
    
    try:
        # Convert the embedding from a NumPy array to a list
        embedding_as_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        vector_db.add(
            embeddings=[embedding_as_list],  # Now this is a list of lists
            metadatas=[metadata],            # List of metadata dictionaries
            ids=[asset_id]                   # List of IDs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving embeddings: {e}")

def load_vector_db():
    client = Client()
    return client.get_or_create_collection(name="documents")