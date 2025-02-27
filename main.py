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
import re

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

def is_greeting(text: str) -> bool:
    """Detect if the text is a greeting or small talk"""
    # Common greeting patterns
    greeting_patterns = [
        r'^hi\b',
        r'^hello\b',
        r'^hey\b',
        r'^greetings\b',
        r'^good\s(morning|afternoon|evening|day)\b',
        r'^how\s(are\syou|is\sit\sgoing)\b',
        r'^what\'?s\sup\b',
        r'^yo\b',
        r'^howdy\b',
        r'^welcome\b',
        r'^nice\sto\smeet\syou\b',
    ]
    
    # Check if any of the patterns match the beginning of the text
    text_lower = text.lower().strip()
    for pattern in greeting_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for short text that might be a greeting (less than 20 chars and no question marks)
    if len(text) < 20 and '?' not in text:
        return True
    
    return False

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
        
        # Check if the query is a greeting
        if is_greeting(query_request.query):
            logger.info("Detected greeting message")
            return {
                "query": query_request.query,
                "response": "Hello! I'm your document assistant. I can help answer questions about the documents you've uploaded. How can I assist you today?",
                "query_type": "greeting",
                "retrieved_documents": []
            }
        
        # If not a greeting, proceed with normal query flow
        query_embedding = get_embedding(query_request.query)
        
        logger.info("Querying Pinecone index")
        results = index.query(
            vector=query_embedding,
            top_k=query_request.top_k,
            include_metadata=True
        )

        # Check if the query is relevant to indexed documents
        if not results.matches or all(match.score < 0.65 for match in results.matches):
            logger.info("Query appears to be unrelated to indexed documents")
            return {
                "query": query_request.query,
                "response": "I appreciate your question, but it appears to be outside the scope of the documents in our knowledge base. I can only provide information based on the documents that have been uploaded. Please consider rephrasing your question to focus on the available content, or upload additional relevant documents if needed.",
                "query_type": "off_topic",
                "retrieved_documents": []
            }

        documents = [match.metadata["text"] for match in results.matches]
        sources = [match.metadata.get("source", "unknown") for match in results.matches]
        scores = [float(match.score) for match in results.matches]

        logger.info("Generating response using OpenAI")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": (
                    "You are a helpful and professional document assistant. "
                    "Your purpose is to provide accurate information from the uploaded documents only. "
                    "If the user's query cannot be sufficiently answered with the provided context: "
                    "1. Acknowledge the question politely "
                    "2. Explain that you can only respond based on information in the uploaded documents "
                    "3. If possible, suggest how they might rephrase their question to get information that is available "
                    "Never make up information or respond with details not present in the context."
                )},
                {"role": "user", "content": f"User Query: {query_request.query}\n\nDocument Context: {' '.join(documents)}"}
            ]
        )

        logger.info("Query processing completed successfully")
        return {
            "query": query_request.query,
            "response": response["choices"][0]["message"]["content"],
            "query_type": "document_query",
            "retrieved_documents": [
                {"text": doc, "source": src, "similarity_score": score} 
                for doc, src, score in zip(documents, sources, scores)
            ]
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
