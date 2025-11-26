from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path
import time
import requests
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import logging
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Company Knowledge Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIR = "./storage"
CHROMA_DIR = "./chroma_db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

Settings.llm = Ollama(model="qwen2:7b-instruct-q4_0", base_url=OLLAMA_BASE_URL, request_timeout=180.0, system_prompt="Odgovaraj isključivo na srpskom jeziku (latinica). " "Ako korisnik koristi drugi jezik, i dalje odgovori na srpskom.",
temperature=0.2,)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",  # multilingual; better for Serbian queries/passages
    cache_folder="./model_cache",
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def wait_for_chromadb(host, port, max_retries=60, delay=2):
    """Wait for ChromaDB to be ready"""
    logger.info(f"Waiting for ChromaDB at {host}:{port}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"http://{host}:{port}/api/v2/heartbeat", timeout=5)
            if response.status_code == 200:
                logger.info(f"ChromaDB is ready after {attempt + 1} attempts!")
                return True
        except Exception as e:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Waiting for ChromaDB...")
            time.sleep(delay)
    
    logger.error(f"ChromaDB did not become ready after {max_retries} attempts")
    raise ConnectionError("Could not connect to ChromaDB")

# Wait for ChromaDB before initializing client
wait_for_chromadb(CHROMA_HOST, CHROMA_PORT)

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Initialize Docling converter
doc_converter = DocumentConverter()

# NOW your existing line:
query_engine = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class DocumentUpload(BaseModel):
    text: str
    metadata: Optional[dict] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    documents_count: Optional[int] = None

def initialize_index():
    global query_engine
    try:
        collection = chroma_client.get_or_create_collection(name="company_docs")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        if collection.count() > 0:
            logger.info(f"Loading existing index with {collection.count()} documents")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new index")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex([], storage_context=storage_context)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        query_engine = index.as_query_engine(similarity_top_k=5)
        logger.info("Query engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing index: {str(e)}")
        return False

def parse_document_with_docling(file_path: str, filename: str) -> List[Document]:
    try:
        logger.info(f"Parsing document with Docling: {filename}")
        
        # Convert document using Docling 2.x API
        result = doc_converter.convert(file_path)
        documents = []
        
        # Extract markdown content
        markdown_content = result.document.export_to_markdown()
        
        if markdown_content:
            # Create main document with full content
            main_doc = Document(
                text=markdown_content,
                metadata={
                    "source": filename,
                    "type": "full_document",
                    "page_count": result.document.num_pages if hasattr(result.document, 'num_pages') else 1
                }
            )
            documents.append(main_doc)
            logger.info(f"Extracted {len(markdown_content)} characters from {filename}")
        
        # Extract tables separately for better retrieval
        if hasattr(result.document, 'tables') and result.document.tables:
            for idx, table in enumerate(result.document.tables):
                # Docling 2.x table export
                if hasattr(table, 'export_to_markdown'):
                    table_text = table.export_to_markdown()
                elif hasattr(table, 'to_markdown'):
                    table_text = table.to_markdown()
                else:
                    table_text = str(table)
                
                table_doc = Document(
                    text=f"Table {idx + 1}:\n{table_text}",
                    metadata={
                        "source": filename,
                        "type": "table",
                        "table_index": idx
                    }
                )
                documents.append(table_doc)
            logger.info(f"Extracted {len(result.document.tables)} tables from {filename}")
        
        # Extract page-level content for better granularity
        if hasattr(result.document, 'pages') and result.document.pages:
            for page_idx, page in enumerate(result.document.pages):
                # Docling 2.x page export
                if hasattr(page, 'export_to_markdown'):
                    page_text = page.export_to_markdown()
                elif hasattr(page, 'to_markdown'):
                    page_text = page.to_markdown()
                elif hasattr(page, 'text'):
                    page_text = page.text
                else:
                    page_text = ""
                
                if page_text and len(page_text.strip()) > 50:
                    page_doc = Document(
                        text=page_text,
                        metadata={
                            "source": filename,
                            "type": "page",
                            "page_number": page_idx + 1
                        }
                    )
                    documents.append(page_doc)
            logger.info(f"Extracted {len(result.document.pages)} pages from {filename}")
        
        return documents
    
    except Exception as e:
        logger.error(f"Error parsing document with Docling: {str(e)}")
        # Fallback to simple text extraction
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                if text:
                    return [Document(
                        text=text,
                        metadata={"source": filename, "type": "fallback"}
                    )]
        except Exception as fallback_error:
            logger.error(f"Fallback parsing also failed: {str(fallback_error)}")
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    success = initialize_index()
    if not success:
        logger.error("Failed to initialize index on startup")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "query_engine_ready": query_engine is not None}

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")
    
    try:
        logger.info(f"Processing query: {request.query}")
        response = query_engine.query(request.query)
        
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if hasattr(node, 'metadata'):
                    source = node.metadata.get('source', 'Unknown')
                    doc_type = node.metadata.get('type', '')
                    page_num = node.metadata.get('page_number', '')
                    
                    source_str = source
                    if page_num:
                        source_str += f" (Page {page_num})"
                    if doc_type == 'table':
                        table_idx = node.metadata.get('table_index', 0)
                        source_str += f" - Table {table_idx + 1}"
                    
                    if source_str not in sources:
                        sources.append(source_str)
        
        return QueryResponse(
            answer=str(response),
            sources=sources if sources else None
        )
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/api/documents/upload", response_model=StatusResponse)
async def upload_document(doc: DocumentUpload):
    global query_engine
    try:
        logger.info("Uploading new text document")
        document = Document(
            text=doc.text,
            metadata=doc.metadata or {"source": "text_upload"}
        )
        
        collection = chroma_client.get_or_create_collection(name="company_docs")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        if collection.count() > 0:
            index = load_index_from_storage(
                StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=PERSIST_DIR
                )
            )
            index.insert(document)
        else:
            index = VectorStoreIndex.from_documents(
                [document],
                storage_context=storage_context
            )
        
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        query_engine = index.as_query_engine(similarity_top_k=5)
        
        return StatusResponse(
            status="success",
            message="Document uploaded successfully",
            documents_count=collection.count()
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/documents/upload-file")
async def upload_file(file: UploadFile = File(...)):
    global query_engine
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file.filename
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Skip Docling for simple text-like files to avoid noisy format errors
            ext = temp_file_path.suffix.lower()
            if ext in {".txt", ".md", ".csv"}:
                with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    documents = [Document(
                        text=f.read(),
                        metadata={"source": file.filename, "type": "fallback"}
                    )]
            else:
                documents = parse_document_with_docling(str(temp_file_path), file.filename)
            
            if not documents:
                raise HTTPException(status_code=400, detail="No content extracted from document")
            
            collection = chroma_client.get_or_create_collection(name="company_docs")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            if collection.count() > 0:
                index = load_index_from_storage(
                    StorageContext.from_defaults(
                        vector_store=vector_store,
                        persist_dir=PERSIST_DIR
                    )
                )
                for doc in documents:
                    index.insert(doc)
            else:
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context
                )
            
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            query_engine = index.as_query_engine(similarity_top_k=5)
            
            logger.info(f"Successfully processed {file.filename}: {len(documents)} document chunks created")
            
            return StatusResponse(
                status="success",
                message=f"File uploaded and parsed successfully. Created {len(documents)} document chunks.",
                documents_count=collection.count()
            )
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/api/documents/count")
async def get_document_count():
    try:
        collection = chroma_client.get_collection(name="company_docs")
        return {"count": collection.count()}
    except Exception as e:
        return {"count": 0}

@app.delete("/api/documents/clear")
async def clear_documents():
    global query_engine
    try:
        chroma_client.delete_collection(name="company_docs")
        initialize_index()
        return StatusResponse(
            status="success",
            message="All documents cleared",
            documents_count=0
        )
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
