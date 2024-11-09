import os
import fitz  # PyMuPDF for PDF reading
from docx import Document  # python-docx for DOCX reading
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException, UploadFile
from utils1 import generate_asset_id, save_embeddings, load_vector_db
from models1 import DocumentProcessResponse

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

vector_db = load_vector_db()

class DocumentProcessingService:
    async def process_document(self, file: UploadFile) -> DocumentProcessResponse:
        # Generate a unique asset ID
        asset_id = generate_asset_id(file.filename)

        # Determine file type and extract content accordingly
        if file.filename.endswith('.txt'):
            content = await file.read()
            content = content.decode('utf-8')

        elif file.filename.endswith('.pdf'):
            content = self.extract_text_from_pdf(file)

        elif file.filename.endswith('.docx'):
            content = self.extract_text_from_docx(file)

        else:
            raise HTTPException(status_code=400, detail="File is not a valid text, PDF, or DOCX file")

        # Generate embedding using sentence-transformers
        embedding = model.encode(content)

        # Save the embedding in the vector database
        save_embeddings(embedding, {'asset_id': asset_id, 'file_name': file.filename}, vector_db)

        return DocumentProcessResponse(asset_id=asset_id)

    def extract_text_from_pdf(self, file: UploadFile) -> str:
        """Extract text content from a PDF file."""
        content = ""
        try:
            # Open the PDF file
            with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
                for page_num in range(pdf.page_count):
                    page = pdf[page_num]
                    content += page.get_text("text")
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")

    def extract_text_from_docx(self, file: UploadFile) -> str:
        """Extract text content from a DOCX file."""
        try:
            # Read the DOCX file
            doc = Document(file.file)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing DOCX file: {e}")
