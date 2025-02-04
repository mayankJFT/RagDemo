from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
from io import BytesIO
import uuid
import logging
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Load and validate API keys
def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Missing environment variable: {var_name}")
    return value

# Initialize API keys and services
try:
    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
    PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = 'us-east-1'
except ValueError as e:
    logger.error(f"Configuration Error: {e}")
    raise

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
try:
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT
    )
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

INDEX_NAME = "rag-index"

# Initialize Pinecone index
try:
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Failed to setup Pinecone index: {e}")
    raise

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text content from PDF file"""
    try:
        logger.info("Starting PDF text extraction")
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            logger.info(f"Processing page {page_num + 1}")
            text += page.extract_text() + "\n"
        logger.info("PDF text extraction completed")
        return text
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into smaller chunks"""
    try:
        logger.info(f"Starting text chunking with chunk size {chunk_size}")
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Text chunking completed. Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error during text chunking: {e}")
        raise

def get_embedding(text: str) -> list:
    """Get OpenAI embedding for text"""
    try:
        logger.info("Generating embedding")
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        logger.info("Embedding generated successfully")
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

async def process_single_pdf(file: UploadFile) -> dict:
    """Process a single PDF file and return its processing results"""
    try:
        logger.info(f"Processing PDF: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type uploaded: {file.filename}")
            return {
                "filename": file.filename,
                "status": "error",
                "message": "File must be a PDF"
            }
        
        # Read the PDF file
        pdf_content = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_content)
        
        if not text.strip():
            logger.warning(f"Extracted text is empty for {file.filename}")
            return {
                "filename": file.filename,
                "status": "error",
                "message": "No text could be extracted from the PDF"
            }
        
        # Split text into chunks
        chunks = chunk_text(text)
        
        # Process each chunk and upload to Pinecone
        uploaded_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} for {file.filename}")
            chunk_id = f"{file.filename}-{uuid.uuid4()}"
            embedding = get_embedding(chunk)
            
            # Upload to Pinecone
            index.upsert(vectors=[(chunk_id, embedding, {"text": chunk, "source": file.filename})])
            uploaded_chunks.append(chunk_id)
        
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_processed": len(uploaded_chunks),
            "chunk_ids": uploaded_chunks
        }
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "status": "error",
            "message": str(e)
        }

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple PDF files concurrently.
    Returns processing results for each file.
    """
    try:
        logger.info(f"Starting batch PDF upload process for {len(files)} files")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
            
        # Process files concurrently
        tasks = [process_single_pdf(file) for file in files]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        return {
            "message": f"Processed {len(files)} PDFs",
            "total_files": len(files),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Unexpected error during batch PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {str(e)}")

@app.post("/query/")
async def query_docs(query_request: QueryRequest):
    """Query the vector store and get relevant responses"""
    try:
        logger.info(f"Processing query: {query_request.query}")
        query_embedding = get_embedding(query_request.query)
        
        logger.info("Querying Pinecone index")
        results = index.query(
            vector=query_embedding,
            top_k=query_request.top_k,
            include_metadata=True
        )

        documents = [match.metadata["text"] for match in results.matches]
        sources = [match.metadata.get("source", "unknown") for match in results.matches]

        logger.info("Generating response using OpenAI")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
           messages=[
                {"role": "system", "content": "You are a helpful assistant. Greet the user with a greeting message and only provide information present in the document."},
                {"role": "user", "content": f"{query_request.query}\nContext: {' '.join(documents)}"}
            ]
        )

        logger.info("Query processing completed successfully")
        return {
            "query": query_request.query,
            "response": response["choices"][0]["message"]["content"],
            "retrieved_documents": [{"text": doc, "source": src} for doc, src in zip(documents, sources)]
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "RAG Backend Running", 
        "status": "healthy",
        "services": {
            "openai": True,
            "pinecone": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
