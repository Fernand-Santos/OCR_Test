import streamlit as st
import sqlite3
import hashlib
import os
import tempfile
import json
import subprocess
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA



# Document processing imports
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd

# Constants
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500  # ~300-800 tokens (assuming ~1.5 chars per token)
CHUNK_OVERLAP = 100  # ~20% overlap
TOP_K_DEFAULT = 5
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for retrieval

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_content_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash of file bytes for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()

def get_file_size_mb(file_bytes: bytes) -> float:
    """Get file size in MB."""
    return len(file_bytes) / (1024 * 1024)

# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================

def check_ollama_status(base_url: str) -> Tuple[bool, str]:
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "✅ Connected"
    except requests.exceptions.ConnectionError:
        return False, "❌ Connection Failed"
    except requests.exceptions.Timeout:
        return False, "⏱️ Timeout"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

    return False, "❌ Unknown Error"

def list_ollama_models(base_url: str) -> List[str]:
    """List available Ollama models via subprocess or API fallback."""
    try:
        # Try subprocess first (preferred)
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Skip header
                return [line.split()[0] for line in lines[1:] if line.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to API
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception:
        pass

    return []

def classify_models(models: List[str]) -> Tuple[List[str], List[str]]:
    """Classify models into embedding and chat models."""
    embedding_keywords = ['embed', 'nomic-embed-text', 'mxbai-embed-large']
    embeddings = [m for m in models if any(kw in m.lower() for kw in embedding_keywords)]
    chat = [m for m in models if m not in embeddings]
    return embeddings, chat

# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_pdf_text(file_path: str) -> List[Dict]:
    """Extract text from PDF using PyMuPDF (Fitz) with caching."""
    # Check cache first
    cache_key = f"pdf_{file_path}"
    if cache_key in st.session_state.text_extraction_cache:
        return st.session_state.text_extraction_cache[cache_key]

    pages_data = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        pages_data.append({
            'page_number': page_num + 1,  # 1-based for UI
            'content_type': 'fitz_text',
            'text': text
        })

    doc.close()

    # Cache the result
    st.session_state.text_extraction_cache[cache_key] = pages_data
    return pages_data

def extract_image_text(file_path: str) -> List[Dict]:
    """Extract text from images using OCR (placeholder - DeepSeek integration stub)."""
    # TODO: Integrate with DeepSeek OCR API
    # For now, return placeholder data
    return [{
        'page_number': 1,
        'content_type': 'ocr_markdown',
        'text': f"[OCR Placeholder] Content from {Path(file_path).name} - DeepSeek OCR integration needed"
    }]

# =============================================================================
# SQLITE DATABASE MANAGEMENT
# =============================================================================

def get_table_columns(sqlite_conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of column names for a table using PRAGMA table_info."""
    cursor = sqlite_conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]  # row[1] is the column name
    return columns

def ensure_schema(sqlite_conn: sqlite3.Connection) -> None:
    """Ensure required columns and indices exist, creating them if missing."""
    cursor = sqlite_conn.cursor()
    
    # 1. Inspect current schema
    try:
        columns = get_table_columns(sqlite_conn, "documents")
        logger.debug(f"Current documents table columns: {columns}")
    except sqlite3.OperationalError:
        # Table doesn't exist yet, will be created by init_sqlite_db
        logger.debug("documents table does not exist yet")
        return
    
    # 2. Add content_hash column if missing
    if "content_hash" not in columns:
        logger.info("Adding missing content_hash column to documents table")
        try:
            cursor.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            sqlite_conn.commit()
            logger.info("Successfully added content_hash column")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not add content_hash column: {e}")
            # Continue - will handle gracefully in queries
    
    # 3. Create UNIQUE index on content_hash if it doesn't exist
    try:
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash 
            ON documents(content_hash)
        """)
        sqlite_conn.commit()
        logger.debug("Ensured UNIQUE index on content_hash exists")
    except sqlite3.OperationalError as e:
        logger.warning(f"Could not create index on content_hash: {e}")
        # Index might fail if column doesn't exist, but we'll handle it
    
    # 4. Backfill NULL content_hash values if possible
    # Note: We can't compute hash from existing rows without file bytes,
    # so we'll allow NULLs but prevent new NULLs via application logic
    cursor.execute("SELECT COUNT(*) FROM documents WHERE content_hash IS NULL")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        logger.warning(f"Found {null_count} documents with NULL content_hash (legacy rows)")

def column_exists(sqlite_conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    try:
        columns = get_table_columns(sqlite_conn, table_name)
        return column_name in columns
    except sqlite3.OperationalError:
        return False

def init_sqlite_db() -> sqlite3.Connection:
    """Initialize ephemeral SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT,
            file_name TEXT,
            mime TEXT,
            size_bytes INTEGER,
            created_at TEXT
        )
    """)
    
    # Ensure schema is up to date (adds column/index if needed)
    ensure_schema(conn)
    
    # Now add NOT NULL constraint if column exists (for new databases)
    # Note: We can't add NOT NULL to existing column with NULLs, so we allow NULLs
    # but enforce uniqueness and non-NULL in application logic

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            doc_id INTEGER,
            page_number INTEGER,
            content_type TEXT,
            text TEXT,
            PRIMARY KEY (doc_id, page_number, content_type),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id INTEGER,
            file_name TEXT,
            page_start INTEGER,
            page_end INTEGER,
            chunk_index INTEGER,
            text TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_logs (
            ts TEXT,
            query TEXT,
            embedding_model TEXT,
            chat_model TEXT,
            selected_files TEXT,
            topk INTEGER,
            results_json TEXT
        )
    """)

    return conn

def log_retrieval(sqlite_conn: sqlite3.Connection, query: str, embedding_model: str,
                 chat_model: str, selected_files: List[str], topk: int, results: List[Dict]):
    """Log retrieval operation to SQLite."""
    sqlite_conn.execute("""
        INSERT INTO retrieval_logs (ts, query, embedding_model, chat_model, selected_files, topk, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        query,
        embedding_model,
        chat_model,
        json.dumps(selected_files),
        topk,
        json.dumps(results)
    ))
    sqlite_conn.commit()

# =============================================================================
# CHROMA DB MANAGEMENT
# =============================================================================

def init_chroma_db(temp_dir: str, embedding_function: OllamaEmbeddings) -> Chroma:
    """Initialize ChromaDB vector store with embedding function.
    
    Args:
        temp_dir: Directory to persist ChromaDB data
        embedding_function: OllamaEmbeddings instance to use for embeddings
        
    Returns:
        Chroma vectorstore instance configured with embedding function
    """
    if embedding_function is None:
        raise ValueError("embedding_function is required for Chroma initialization")
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=temp_dir,
        embedding_function=embedding_function
    )

# =============================================================================
# DOCUMENT INGESTION PIPELINE
# =============================================================================

def ingest_document(file_bytes: bytes, file_name: str, temp_dir: str,
                   sqlite_conn: sqlite3.Connection, vectorstore: Chroma,
                   embedding_model: OllamaEmbeddings, batch_size: int = 10) -> int:
    """Complete document ingestion pipeline."""

    # 0. Ensure schema is up to date
    ensure_schema(sqlite_conn)

    # 1. Compute content hash for duplicate detection
    content_hash = compute_content_hash(file_bytes)
    
    # Debug logging
    logger.debug(f"Ingesting document: filename={file_name}, hash={content_hash[:16]}...")

    # 2. Check for existing document with same content hash (defensive check)
    cursor = sqlite_conn.cursor()
    existing_doc_id = None
    
    if column_exists(sqlite_conn, "documents", "content_hash"):
        try:
            cursor.execute("SELECT doc_id FROM documents WHERE content_hash = ?", (content_hash,))
            existing = cursor.fetchone()
            if existing:
                existing_doc_id = existing[0]
                logger.debug(f"Duplicate document detected: filename={file_name}, existing doc_id={existing_doc_id}")
                return existing_doc_id
        except sqlite3.OperationalError as e:
            logger.warning(f"Error checking for duplicate: {e}. Proceeding with insertion.")
    else:
        logger.warning("content_hash column does not exist. Skipping duplicate check.")

    # 3. Save file to temp storage (use content_hash for filename)
    file_path = os.path.join(temp_dir, f"{content_hash}_{file_name}")
    with open(file_path, 'wb') as f:
        f.write(file_bytes)

    # 4. Extract text based on file type
    if file_name.lower().endswith('.pdf'):
        pages_data = extract_pdf_text(file_path)
    else:  # Image files
        pages_data = extract_image_text(file_path)

    # 5. Store document metadata in SQLite (doc_id auto-generated)
    try:
        # Build INSERT statement based on whether content_hash column exists
        if column_exists(sqlite_conn, "documents", "content_hash"):
            cursor.execute("""
                INSERT INTO documents (content_hash, file_name, mime, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content_hash,
                file_name,
                'application/pdf' if file_name.lower().endswith('.pdf') else 'image/jpeg',
                len(file_bytes),
                datetime.now().isoformat()
            ))
        else:
            # Fallback: insert without content_hash (legacy mode)
            logger.warning("Inserting without content_hash column (legacy mode)")
            cursor.execute("""
                INSERT INTO documents (file_name, mime, size_bytes, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                file_name,
                'application/pdf' if file_name.lower().endswith('.pdf') else 'image/jpeg',
                len(file_bytes),
                datetime.now().isoformat()
            ))
        
        # Get the generated doc_id
        doc_id = cursor.lastrowid
        logger.debug(f"Inserted new document: filename={file_name}, doc_id={doc_id}, hash={content_hash[:16]}...")
        
    except sqlite3.IntegrityError as e:
        # Handle race condition or unique constraint violation
        if column_exists(sqlite_conn, "documents", "content_hash"):
            try:
                cursor.execute("SELECT doc_id FROM documents WHERE content_hash = ?", (content_hash,))
                existing = cursor.fetchone()
                if existing:
                    doc_id = existing[0]
                    logger.debug(f"Race condition handled: duplicate detected after insert attempt, doc_id={doc_id}")
                    return doc_id
            except sqlite3.OperationalError:
                pass
        
        # Unexpected integrity error
        logger.error(f"Unexpected IntegrityError: {e}")
        raise
    except sqlite3.OperationalError as e:
        # Handle case where column doesn't exist (shouldn't happen after ensure_schema, but be defensive)
        logger.error(f"OperationalError during insert: {e}")
        raise

    # 6. Store pages in SQLite
    for page_data in pages_data:
        cursor.execute("""
            INSERT INTO pages (doc_id, page_number, content_type, text)
            VALUES (?, ?, ?, ?)
        """, (
            doc_id,
            page_data['page_number'],
            page_data['content_type'],
            page_data['text']
        ))

    sqlite_conn.commit()

    # 7. Chunk the text (300-800 tokens per chunk)
    # CHUNK_SIZE=500 characters ≈ 300-800 tokens (assuming ~1.5 chars per token)
    full_text = "\n".join([p['text'] for p in pages_data])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(full_text)
    
    logger.debug(f"Split document into {len(chunks)} chunks (target: 300-800 tokens per chunk)")

    # 8. Create LangChain documents with metadata - each chunk stored individually
    langchain_docs = []
    for i, chunk in enumerate(chunks):
        # Estimate page range (simplified - could be more sophisticated)
        chunk_start_page = max(1, int((i * len(pages_data)) / len(chunks)) + 1)
        chunk_end_page = min(len(pages_data), chunk_start_page)

        # Each chunk gets unique ID: doc_id:chunk_index
        chunk_id = f"{doc_id}:{i}"
        
        metadata = {
            'doc_id': str(doc_id),  # Convert to string for metadata compatibility
            'file_name': file_name,
            'page_start': chunk_start_page,
            'page_end': chunk_end_page,
            'chunk_index': i,
            'source_type': pages_data[0]['content_type'] if pages_data else 'unknown'
        }

        # Create Document object for this chunk (stored individually)
        langchain_docs.append(Document(page_content=chunk, metadata=metadata))

        # Store chunk in SQLite with unique chunk_id
        cursor.execute("""
            INSERT INTO chunks (chunk_id, doc_id, file_name, page_start, page_end, chunk_index, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            doc_id,
            file_name,
            chunk_start_page,
            chunk_end_page,
            i,
            chunk
        ))

    sqlite_conn.commit()

    # 9. Embed and store each chunk individually in ChromaDB with batching
    # Each chunk gets its own embedding vector stored in the vector database
    if langchain_docs:
        logger.debug(f"Embedding and storing {len(langchain_docs)} chunks in ChromaDB")
        # Process in batches to optimize memory usage
        for i in range(0, len(langchain_docs), batch_size):
            batch = langchain_docs[i:i + batch_size]
            # Each document in batch is a separate chunk with its own embedding
            vectorstore.add_documents(batch)
        logger.debug(f"Successfully stored {len(langchain_docs)} chunk embeddings in ChromaDB")

    return doc_id

# =============================================================================
# RAG QUERYING
# =============================================================================

def query_documents(query: str, vectorstore: Chroma, chat_model: ChatOllama,
                   selected_files: List[str], top_k: int,
                   sqlite_conn: sqlite3.Connection, embedding_model_name: str,
                   chat_model_name: str) -> Tuple[str, List[Dict]]:
    """Execute RAG query with semantic retrieval, similarity ranking, and grounded prompting.
    
    Note: Chroma handles query embedding internally via its embedding_function.
    This ensures consistent embedding model usage across ingestion and query.
    """
    
    # Ensure top_k is at least 5
    top_k = max(5, top_k)
    
    logger.info(f"Query: {query}")
    logger.info(f"Top-K: {top_k}, Selected files: {selected_files}")
    
    # 1. Build filter for selected files (only if not "All")
    # Default behavior: search across ALL documents unless explicitly filtered
    filter_dict = None
    if selected_files and "All" not in selected_files and len(selected_files) > 0:
        filter_dict = {"file_name": {"$in": selected_files}}
        logger.info(f"Applying file filter: {selected_files}")
    else:
        logger.info("Searching across ALL documents (no filter)")
    
    # 2. Perform semantic similarity search with scores
    # Chroma handles query embedding internally using its embedding_function
    # This ensures the same embedding model is used for query as was used for ingestion
    if filter_dict:
        # Use retriever with filter for better performance
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": filter_dict
            }
        )
        # Get documents with scores using retriever
        results_with_scores = vectorstore.similarity_search_with_score(
            query,
            k=top_k,
            filter=filter_dict
        )
    else:
        # No filter - search across ALL documents
        results_with_scores = vectorstore.similarity_search_with_score(
            query,
            k=top_k
        )
    
    # 4. Check if we have any results
    if not results_with_scores:
        logger.warning("No chunks retrieved for query")
        return "No relevant information found.", []
    
    # 5. Extract best similarity score (first result has highest score)
    best_score = results_with_scores[0][1] if results_with_scores else 0.0
    
    # Note: ChromaDB returns distance scores (lower is better), so we need to convert to similarity
    # For cosine similarity: similarity = 1 - distance (if distance is normalized)
    # For now, assume scores are distances and convert: similarity = 1 / (1 + distance)
    # Or if using cosine: similarity = 1 - distance
    # Let's check the score type - Chroma typically uses cosine distance
    # Convert distance to similarity score (assuming cosine distance where 0 = identical, 2 = opposite)
    similarity_scores = []
    for doc, distance in results_with_scores:
        # Convert distance to similarity (cosine distance: 0 = identical, 2 = opposite)
        # Similarity = 1 - (distance / 2) for cosine distance
        # Or use: similarity = max(0, 1 - distance) if distance is normalized [0, 2]
        similarity = max(0.0, 1.0 - (distance / 2.0)) if distance <= 2.0 else 1.0 / (1.0 + distance)
        similarity_scores.append(similarity)
    
    best_similarity = similarity_scores[0] if similarity_scores else 0.0
    
    # 6. Check threshold
    if best_similarity < SIMILARITY_THRESHOLD:
        logger.warning(f"Best similarity score {best_similarity:.4f} below threshold {SIMILARITY_THRESHOLD}")
        return "No relevant information found.", []
    
    # 7. Sort by similarity (already sorted by ChromaDB, but ensure descending order)
    # Pair documents with their similarity scores and sort
    doc_score_pairs = list(zip([doc for doc, _ in results_with_scores], similarity_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity descending
    
    # 8. Extract chunks and metadata
    retrieved_chunks = []
    chunk_ids = []
    for i, (doc, similarity) in enumerate(doc_score_pairs):
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        page_content = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
        
        chunk_id = f"{metadata.get('doc_id', 'unknown')}:{metadata.get('chunk_index', i)}"
        chunk_ids.append(chunk_id)
        
        retrieved_chunks.append({
            'chunk_id': chunk_id,
            'file_name': metadata.get('file_name', 'Unknown'),
            'page_start': metadata.get('page_start', 1),
            'page_end': metadata.get('page_end', 1),
            'chunk_index': metadata.get('chunk_index', 0),
            'content': page_content,
            'similarity_score': similarity
        })
    
    # 9. Log retrieval details (diagnostics)
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks across {len(set(chunk['file_name'] for chunk in retrieved_chunks))} document(s)")
    logger.info(f"Chunk IDs: {chunk_ids}")
    for i, chunk in enumerate(retrieved_chunks):
        logger.info(f"  Chunk {i+1}: file_name={chunk['file_name']}, chunk_index={chunk['chunk_index']}, similarity={chunk['similarity_score']:.4f}")
    logger.info(f"Similarity scores: {[f'{s:.4f}' for s in similarity_scores]}")
    logger.info(f"Best similarity: {best_similarity:.4f}")
    
    # 10. Build grounded prompt
    context_text = "\n\n".join([
        f"[Chunk {i+1} from {chunk['file_name']}]: {chunk['content']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    grounded_prompt = f"""Answer ONLY from the following context. If the answer is not found in the context, say "NOT FOUND."

Context:
{context_text}

Question: {query}

Answer:"""
    
    # 11. Generate answer using LLM
    try:
        response = chat_model.invoke(grounded_prompt)
        # Extract text from response (handle different response formats)
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer = f"Error generating answer: {str(e)}"
    
    # 12. Prepare source documents for logging
    source_docs = [{
        'file_name': chunk['file_name'],
        'page_start': chunk['page_start'],
        'page_end': chunk['page_end'],
        'chunk_index': chunk['chunk_index'],
        'content': chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'],
        'similarity_score': chunk['similarity_score']
    } for chunk in retrieved_chunks]
    
    # 13. Log retrieval to SQLite
    log_retrieval(sqlite_conn, query, embedding_model_name, chat_model_name,
                 selected_files, top_k, source_docs)
    
    return answer, source_docs

# =============================================================================
# DOCUMENT DELETION
# =============================================================================

def delete_document(doc_id: int, sqlite_conn: sqlite3.Connection,
                   vectorstore: Chroma, docs_index: Dict) -> bool:
    """Delete document from all storage locations."""

    try:
        # Remove from SQLite
        sqlite_conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        sqlite_conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
        sqlite_conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        sqlite_conn.commit()

        # Remove from ChromaDB (convert to string for metadata compatibility)
        vectorstore.delete(where={"doc_id": str(doc_id)})

        # Remove temp file
        if doc_id in docs_index:
            file_path = docs_index[doc_id].get('stored_path')
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            del docs_index[doc_id]

        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'sqlite_conn' not in st.session_state:
        st.session_state.sqlite_conn = init_sqlite_db()
        # Ensure schema is up to date (handles migration scenarios)
        ensure_schema(st.session_state.sqlite_conn)

    if 'temp_dir_path' not in st.session_state:
        st.session_state.temp_dir_path = tempfile.mkdtemp(prefix="doc_proc_")

    # Chroma initialization is deferred until embedding model is selected
    # This ensures consistent embedding model usage
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = None

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    if 'embedding_model_instance' not in st.session_state:
        st.session_state.embedding_model_instance = None

    if 'ollama_models_cache' not in st.session_state:
        st.session_state.ollama_models_cache = []

    if 'embedding_model_name' not in st.session_state:
        st.session_state.embedding_model_name = ""

    if 'chat_model_name' not in st.session_state:
        st.session_state.chat_model_name = ""

    if 'docs_index' not in st.session_state:
        st.session_state.docs_index = {}

    if 'last_retrieval_hits' not in st.session_state:
        st.session_state.last_retrieval_hits = []
    
    if 'last_uploaded_doc_id' not in st.session_state:
        st.session_state.last_uploaded_doc_id = None

    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False

    if 'viewer_expanded' not in st.session_state:
        st.session_state.viewer_expanded = False

    # Caching for performance optimization
    if 'processed_docs_cache' not in st.session_state:
        st.session_state.processed_docs_cache = {}  # doc_id -> processing metadata

    if 'text_extraction_cache' not in st.session_state:
        st.session_state.text_extraction_cache = {}  # file_path -> extracted_pages

    if 'ollama_models_cache_timestamp' not in st.session_state:
        st.session_state.ollama_models_cache_timestamp = 0

    if 'show_full_text_modal' not in st.session_state:
        st.session_state.show_full_text_modal = False

    if 'selected_doc_for_text' not in st.session_state:
        st.session_state.selected_doc_for_text = None

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_document_viewer():
    """Render document viewer panel on the right side taking up half the page."""
    # Add custom CSS for the document viewer
    st.markdown("""
    <style>
    .doc-image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    
    .doc-image-container:hover {
        transform: scale(1.02);
    }
    
    .doc-info-badge {
        display: inline-block;
        padding: 6px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin: 5px 0;
    }
    
    /* Auto-fit image container */
    .doc-image-container img {
        width: 100%;
        height: auto;
        object-fit: contain;
        border-radius: 8px;
    }
    
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Document viewer header with expand/collapse toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📄 Original Document Viewer")
    with col2:
        expand_icon = "🔽" if not st.session_state.viewer_expanded else "🔼"
        if st.button(f"{expand_icon} Expand", key="viewer_toggle",
                    help="Toggle viewer expansion"):
            st.session_state.viewer_expanded = not st.session_state.viewer_expanded
            st.rerun()
    
    if not st.session_state.docs_index:
        st.info("📝 Upload documents to view them here")
        return
    
    # Create list of documents
    doc_list = list(st.session_state.docs_index.items())
    
    if not doc_list:
        st.info("📝 No documents available to view")
        return
    
    # File selector with auto-selection of last uploaded document
    file_options = {doc_info['file_name']: doc_id for doc_id, doc_info in doc_list}
    file_names = list(file_options.keys())
    
    # Auto-select last uploaded document if available
    default_index = 0
    if st.session_state.last_uploaded_doc_id and st.session_state.last_uploaded_doc_id in file_options.values():
        # Find the index of the last uploaded document
        for idx, (name, doc_id) in enumerate(file_options.items()):
            if doc_id == st.session_state.last_uploaded_doc_id:
                default_index = idx
                break
    
    selected_file = st.selectbox(
        "Choose document to view:",
        options=file_names,
        index=default_index,
        key="doc_viewer_select",
        label_visibility="visible"
    )

    if selected_file:
        selected_doc_id = file_options[selected_file]

        # Show full text button after selected_doc_id is defined
        col1, col2 = st.columns([1, 1])
        with col1:
            show_text = st.button("📖 Show Full Text", key=f"show_text_{selected_doc_id}")
        with col2:
            pass

        if show_text:
            st.session_state.show_full_text_modal = True
            st.session_state.selected_doc_for_text = selected_doc_id
            st.rerun()
        doc_info = st.session_state.docs_index[selected_doc_id]
        file_path = doc_info['stored_path']
        
        if os.path.exists(file_path):
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                # Display PDF pages as images
                try:
                    doc = fitz.open(file_path)
                    num_pages = len(doc)
                    
                    st.markdown(f"""
                    <div class="doc-info-badge">
                        📄 {selected_file} • {num_pages} page{'s' if num_pages > 1 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Page selector for multi-page PDFs
                    if num_pages > 1:
                        # Initialize session state for page number
                        page_key = f"pdf_page_{selected_doc_id}"
                        if page_key not in st.session_state:
                            st.session_state[page_key] = 1

                        # Combined page navigation: slider and direct input
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            page_num = st.slider(
                                "Navigate pages:",
                                min_value=1,
                                max_value=num_pages,
                                value=st.session_state[page_key],
                                key=f"pdf_slider_{selected_doc_id}",
                                label_visibility="visible"
                            )
                        with col2:
                            # Direct page input with Enter key support
                            direct_page = st.number_input(
                                "Go to page:",
                                min_value=1,
                                max_value=num_pages,
                                value=st.session_state[page_key],
                                step=1,
                                key=f"pdf_input_{selected_doc_id}",
                                label_visibility="visible"
                            )
                        with col3:
                            if st.button("Go", key=f"go_page_{selected_doc_id}"):
                                page_num = direct_page

                        # Update page number from either slider or input
                        if direct_page != st.session_state[page_key]:
                            page_num = direct_page

                        st.session_state[page_key] = page_num
                    else:
                        page_num = 1
                    
                    # Render selected page
                    page = doc.load_page(page_num - 1)
                    # Convert page to image (zoom factor for better quality)
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    st.markdown('<div class="doc-image-container">', unsafe_allow_html=True)
                    st.image(img_data, caption=f"Page {page_num} of {num_pages}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    doc.close()
                except Exception as e:
                    st.error(f"Error displaying PDF: {str(e)}")
            
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                # Display image directly
                try:
                    st.markdown(f"""
                    <div class="doc-info-badge">
                        🖼️ {selected_file}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="doc-image-container">', unsafe_allow_html=True)
                    st.image(file_path, caption=selected_file, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
            else:
                st.warning(f"Unsupported file type: {file_ext}")
        else:
            st.error(f"File not found: {file_path}")

def render_sidebar():
    """Render sidebar controls."""
    # Sidebar collapse/expand toggle
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.sidebar.title("⚙️ Settings")
    with col2:
        if st.sidebar.button("◁" if not st.session_state.sidebar_collapsed else "▷",
                           key="sidebar_toggle", help="Toggle sidebar"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()

    if st.session_state.sidebar_collapsed:
        return None, None, None

    # Ollama Configuration
    st.sidebar.subheader("🤖 Ollama Server")
    ollama_base_url = st.sidebar.text_input(
        "Base URL",
        value=DEFAULT_OLLAMA_BASE_URL,
        help="Default: http://localhost:11434"
    )

    # Server status check
    if st.sidebar.button("🔍 Check Server Status"):
        is_running, status_msg = check_ollama_status(ollama_base_url)
        if is_running:
            st.sidebar.success(status_msg)
        else:
            st.sidebar.error(status_msg)

    # Always show current status
    is_running, status_msg = check_ollama_status(ollama_base_url)
    if is_running:
        st.sidebar.success(f"Server Status: {status_msg}")
    else:
        st.sidebar.error(f"Server Status: {status_msg}")

    # Model Management
    st.sidebar.subheader("🧠 AI Models")

    # Refresh models button
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        refresh_clicked = st.button("🔄 Refresh Models", use_container_width=True)
    with col2:
        if st.session_state.ollama_models_cache:
            st.metric("Models", len(st.session_state.ollama_models_cache))

    if refresh_clicked:
        st.session_state.ollama_models_cache = list_ollama_models(ollama_base_url)
        st.session_state.ollama_models_cache_timestamp = datetime.now().timestamp()
        st.rerun()

    # Display available models
    if st.session_state.ollama_models_cache:
        st.sidebar.subheader("📋 Downloaded Models")
        embedding_models, chat_models = classify_models(st.session_state.ollama_models_cache)

        with st.sidebar.expander("🔍 Embedding Models", expanded=False):
            if embedding_models:
                for model in embedding_models:
                    st.write(f"• {model}")
            else:
                st.write("No embedding models found")

        with st.sidebar.expander("💬 Chat Models", expanded=False):
            if chat_models:
                for model in chat_models:
                    st.write(f"• {model}")
            else:
                st.write("No chat models found")
    else:
        st.sidebar.info("Click 'Refresh Models' to load available models")

    # Model selection
    embedding_models, chat_models = classify_models(st.session_state.ollama_models_cache)

    if embedding_models:
        default_embed = "nomic-embed-text" if "nomic-embed-text" in embedding_models else embedding_models[0]
        selected_embedding = st.sidebar.selectbox(
            "🎯 Embedding Model",
            embedding_models,
            index=embedding_models.index(default_embed) if default_embed in embedding_models else 0
        )
        st.session_state.embedding_model_name = selected_embedding
        
        # Initialize/reinitialize Chroma with embedding model when model is selected/changed
        # This ensures consistent embedding model usage across ingestion and query
        if (st.session_state.embedding_model_instance is None or 
            st.session_state.embedding_model_instance.model != selected_embedding):
            # Create embedding model instance
            st.session_state.embedding_model_instance = OllamaEmbeddings(
                model=selected_embedding,
                base_url=ollama_base_url
            )
            # Initialize Chroma with embedding function
            if st.session_state.vectorstore is None:
                st.session_state.chroma_client = init_chroma_db(
                    st.session_state.temp_dir_path,
                    st.session_state.embedding_model_instance
                )
                st.session_state.vectorstore = st.session_state.chroma_client
            else:
                # Reinitialize if model changed (this will create a new collection)
                # Note: In production, you might want to migrate existing data
                logger.info(f"Reinitializing Chroma with new embedding model: {selected_embedding}")
                st.session_state.chroma_client = init_chroma_db(
                    st.session_state.temp_dir_path,
                    st.session_state.embedding_model_instance
                )
                st.session_state.vectorstore = st.session_state.chroma_client
    else:
        st.sidebar.warning("No embedding models available. Please pull models like 'nomic-embed-text'")
        st.session_state.embedding_model_name = ""
        st.session_state.embedding_model_instance = None

    if chat_models:
        selected_chat = st.sidebar.selectbox("🤖 Chat Model", chat_models)
        st.session_state.chat_model_name = selected_chat
    else:
        st.sidebar.warning("No chat models available. Please pull models like 'qwen2.5' or 'mistral'")
        st.session_state.chat_model_name = ""

    # Retrieval settings
    st.sidebar.subheader("🔍 Retrieval Settings")
    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=TOP_K_DEFAULT)

    # File filter
    stored_files = ["All"] + [doc['file_name'] for doc in st.session_state.docs_index.values()]
    selected_files = st.sidebar.multiselect(
        "📂 Filter by Files",
        stored_files,
        default=["All"]
    )

    if st.session_state.sidebar_collapsed:
        return None, None, None

    return ollama_base_url, top_k, selected_files

def render_full_text_modal():
    """Render modal to display entire OCR'd text content."""
    if st.session_state.show_full_text_modal and st.session_state.selected_doc_for_text:
        doc_id = st.session_state.selected_doc_for_text
        doc_info = st.session_state.docs_index.get(doc_id)

        if doc_info:
            with st.container():
                st.subheader(f"📄 Full Text: {doc_info['file_name']}")

                # Get full text from SQLite
                cursor = st.session_state.sqlite_conn.cursor()
                cursor.execute("SELECT page_number, text FROM pages WHERE doc_id = ? ORDER BY page_number",
                             (doc_id,))
                pages_data = cursor.fetchall()

                if pages_data:
                    full_text = ""
                    for page_num, text in pages_data:
                        full_text += f"\n--- Page {page_num} ---\n{text}\n"

                    # Display in a scrollable text area
                    st.text_area(
                        "Full Document Text",
                        value=full_text,
                        height=400,
                        disabled=True,
                        key=f"full_text_{doc_id}"
                    )

                    # Word count and page count
                    total_words = len(full_text.split())
                    total_pages = len(pages_data)
                    st.caption(f"📊 {total_pages} pages • {total_words:,} words")
                else:
                    st.warning("No text data available for this document.")

                # Close button
                if st.button("❌ Close", key=f"close_text_modal_{doc_id}"):
                    st.session_state.show_full_text_modal = False
                    st.session_state.selected_doc_for_text = None
                    st.rerun()

def render_sqlite_viewer():
    """Render SQLite database viewer."""
    with st.expander("SQLite Storage Viewer", expanded=False):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Documents", "Pages", "Chunks", "Retrieval Logs", "Last Retrieval"])

        with tab1:
            df_docs = pd.read_sql_query("SELECT * FROM documents", st.session_state.sqlite_conn)
            st.dataframe(df_docs)

        with tab2:
            df_pages = pd.read_sql_query("SELECT * FROM pages LIMIT 100", st.session_state.sqlite_conn)
            st.dataframe(df_pages)

        with tab3:
            # Add file filter for chunks
            cursor = st.session_state.sqlite_conn.cursor()
            cursor.execute("SELECT DISTINCT file_name FROM chunks")
            available_files = [row[0] for row in cursor.fetchall()]

            if available_files:
                chunk_filter = st.selectbox("Filter chunks by file", ["All"] + available_files)
                if chunk_filter == "All":
                    df_chunks = pd.read_sql_query("SELECT * FROM chunks LIMIT 100", st.session_state.sqlite_conn)
                else:
                    df_chunks = pd.read_sql_query(
                        "SELECT * FROM chunks WHERE file_name = ? LIMIT 100",
                        (chunk_filter,),
                        st.session_state.sqlite_conn
                    )
                st.dataframe(df_chunks)
            else:
                st.write("No chunks available")

        with tab4:
            df_logs = pd.read_sql_query("SELECT * FROM retrieval_logs ORDER BY ts DESC LIMIT 10", st.session_state.sqlite_conn)
            st.dataframe(df_logs)

        with tab5:
            if st.session_state.last_retrieval_hits:
                df_last = pd.DataFrame(st.session_state.last_retrieval_hits)
                st.dataframe(df_last)
            else:
                st.write("No recent retrievals")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Add global CSS for better UI
    st.markdown("""
    <style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Responsive column layout */
    @media (max-width: 768px) {
        .stColumn {
            width: 100% !important;
        }
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.2s ease;
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Improved selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Better expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Scrollbar styling */
    .element-container {
        scrollbar-width: thin;
    }

    /* Smooth layout transitions */
    .main .block-container,
    .stColumn {
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📄 Document RAG with Ollama")
    st.markdown("Upload PDFs/images, extract text, chunk, embed, and ask questions using Ollama models.")

    # Initialize session state
    init_session_state()

    # Status overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Documents Stored", len(st.session_state.docs_index))
    with col2:
        models_count = len(st.session_state.ollama_models_cache)
        st.metric("🤖 Ollama Models", models_count if models_count > 0 else "Not loaded")
    with col3:
        is_running, _ = check_ollama_status(DEFAULT_OLLAMA_BASE_URL)
        st.metric("🔗 Server Status", "Connected" if is_running else "Disconnected")

    st.markdown("---")

    # Sidebar
    sidebar_result = render_sidebar()
    if sidebar_result is None or len(sidebar_result) != 3:
        # Sidebar collapsed, use default values
        ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        top_k = TOP_K_DEFAULT
        selected_files = ["All"]
    else:
        ollama_base_url, top_k, selected_files = sidebar_result

    # Show setup guidance if needed
    is_running, status_msg = check_ollama_status(ollama_base_url)
    if not is_running or not st.session_state.ollama_models_cache:
        st.warning("🚨 Ollama Setup Required")
        st.markdown("""
        **To get started:**

        1. **Install Ollama** (if not already installed):
           ```bash
           # Download from: https://ollama.ai/
           ```

        2. **Start Ollama server**:
           ```bash
           ollama serve
           ```

        3. **Pull required models**:
           ```bash
           ollama pull nomic-embed-text  # For embeddings
           ollama pull qwen2.5           # For chat (or mistral, llama2, etc.)
           ```

        4. **Check server status** and **refresh models** in the sidebar above.
        """)

        if st.button("🔍 I've completed setup - Check Status"):
            st.rerun()

        st.markdown("---")
        return

    # File Upload Section
    st.header("📤 Document Ingestion")

    # Upload area - clean design without white boxes
    uploaded_files = st.file_uploader(
        "📁 Drag and drop your files here (PDF, JPG, JPEG, PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="visible"
    )

    if uploaded_files:
        st.info(f"✅ {len(uploaded_files)} file(s) selected")
        for file in uploaded_files:
            st.caption(f"• {file.name} ({get_file_size_mb(file.getbuffer()):.2f} MB)")

    if uploaded_files and st.session_state.embedding_model_name:
        # Process button with better styling
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button(
                f"🚀 Process {len(uploaded_files)} Document{'s' if len(uploaded_files) > 1 else ''}",
                use_container_width=True,
                type="primary"
            )

        if process_button:
            with st.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_container = st.empty()

                with status_container.container():
                    st.subheader("📊 Processing Progress")

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"🔄 Processing {file.name}...")

                    # Read file bytes
                    file_bytes = file.getbuffer()

                    # Use the same embedding model instance that was used to initialize Chroma
                    # This ensures consistent embeddings across ingestion and query
                    if st.session_state.embedding_model_instance is None:
                        st.error("Please select an embedding model in the sidebar first.")
                        continue
                    
                    embeddings = st.session_state.embedding_model_instance

                    # Ingest document with batching
                    doc_id = ingest_document(
                        file_bytes, file.name,
                        st.session_state.temp_dir_path,
                        st.session_state.sqlite_conn,
                        st.session_state.vectorstore,
                        embeddings,
                        batch_size=20  # Process 20 documents at a time for optimal performance
                    )

                    # Get content hash for file path
                    content_hash = compute_content_hash(file_bytes)
                    
                    # Update docs index
                    st.session_state.docs_index[doc_id] = {
                        'file_name': file.name,
                        'stored_path': os.path.join(st.session_state.temp_dir_path, f"{content_hash}_{file.name}"),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Track last uploaded document for auto-selection in viewer
                    st.session_state.last_uploaded_doc_id = doc_id

                    progress_bar.progress((i + 1) / len(uploaded_files))

                    # Update status container
                    with status_container.container():
                        st.subheader("📊 Processing Progress")
                        completed = i + 1
                        st.write(f"✅ Completed: {completed}/{len(uploaded_files)} files")
                        st.write(f"📄 Latest: {file.name} → ID: {doc_id}")

                status_text.empty()
                status_container.empty()
                st.success(f"✅ Successfully processed {len(uploaded_files)} document{'s' if len(uploaded_files) > 1 else ''}!")
                st.balloons()
                st.rerun()
    elif uploaded_files and not st.session_state.embedding_model_name:
        st.warning("⚠️ Please select an embedding model in the sidebar before processing documents.")

    # Main content layout - adjust based on sidebar and viewer state
    if st.session_state.viewer_expanded:
        # When viewer is expanded, show only the viewer
        viewer_col = st.container()
        main_col = None
    elif st.session_state.sidebar_collapsed:
        # When sidebar is collapsed, give more space to main content
        main_col, viewer_col = st.columns([2, 1])
    else:
        # Default layout with equal columns
        main_col, viewer_col = st.columns([1, 1])
    
    if main_col is not None:
        with main_col:
            # Document Management
            st.header("📋 Stored Documents")
            if st.session_state.docs_index:
                for doc_id, doc_info in st.session_state.docs_index.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {doc_info['file_name']}")
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{doc_id}"):
                            if delete_document(doc_id, st.session_state.sqlite_conn,
                                             st.session_state.vectorstore, st.session_state.docs_index):
                                st.success(f"Deleted {doc_info['file_name']}")
                                st.rerun()
            else:
                st.info("No documents stored yet.")

            # RAG Query Interface
            st.header("❓ Ask Questions About Your Documents")

            if not st.session_state.docs_index:
                st.info("📝 Upload and process some documents first to start asking questions.")
            else:
                query = st.text_area(
                    "Enter your question:",
                    placeholder="e.g., What are the main findings in the report? or Summarize the key points from the contract...",
                    height=100
                )

                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    search_button = st.button(
                        "🔍 Search & Answer",
                        use_container_width=True,
                        type="primary",
                        disabled=not query.strip()
                    )

                if query and st.session_state.chat_model_name and st.session_state.embedding_model_name and search_button:
                    # Ensure vectorstore is initialized
                    if st.session_state.vectorstore is None:
                        st.error("Please select an embedding model in the sidebar first.")
                    else:
                        with st.spinner("Searching and generating answer..."):
                            # Initialize chat model
                            chat_model = ChatOllama(
                                model=st.session_state.chat_model_name,
                                base_url=ollama_base_url,
                                temperature=0
                            )
                            
                            # Execute query - Chroma handles query embedding internally
                            # using the same embedding_function used during ingestion
                            answer, sources = query_documents(
                                query,
                                st.session_state.vectorstore,
                                chat_model,
                                selected_files,
                                top_k,
                                st.session_state.sqlite_conn,
                                st.session_state.embedding_model_name,
                                st.session_state.chat_model_name
                            )

                        # Store for SQLite viewer
                        st.session_state.last_retrieval_hits = sources

                        # Display results
                        st.subheader("💡 Answer")
                        st.write(answer)

                        st.subheader("📚 Sources")
                        for i, source in enumerate(sources):
                            similarity_score = source.get('similarity_score', 0.0)
                            with st.expander(f"Source {i+1}: {source['file_name']} (Pages {source['page_start']}-{source['page_end']}, Chunk {source['chunk_index']}, Similarity: {similarity_score:.4f})"):
                                st.write(source['content'])

            # SQLite Viewer
            render_sqlite_viewer()

    # Document Viewer - takes up available space
    with viewer_col:
        render_document_viewer()

    # Full text modal
    render_full_text_modal()

# =============================================================================
# REGRESSION TEST
# =============================================================================

def test_duplicate_ingestion():
    """Test that ingesting the same file twice creates only one row and raises no exception."""
    import tempfile
    import shutil
    
    # Create temporary directory
    test_temp_dir = tempfile.mkdtemp(prefix="test_doc_proc_")
    
    try:
        # Initialize database
        test_conn = init_sqlite_db()
        
        # Create mock embeddings
        class MockEmbeddings:
            def embed_documents(self, texts):
                # Return mock embeddings (384 dimensions, common embedding size)
                return [[0.1] * 384 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 384
        
        mock_embeddings = MockEmbeddings()
        
        # Initialize vectorstore with embedding function
        test_vectorstore = init_chroma_db(test_temp_dir, mock_embeddings)
        
        # Create a simple test PDF file (minimal PDF structure)
        # For testing, we'll create a simple text file and mock the extraction
        test_content = b"This is a test document for duplicate ingestion testing."
        test_filename = "test_document.pdf"
        
        # Save test file
        test_file_path = os.path.join(test_temp_dir, "test_input.pdf")
        with open(test_file_path, 'wb') as f:
            f.write(test_content)
        
        # First ingestion (use mock_embeddings)
        doc_id_1 = ingest_document(
            test_content,
            test_filename,
            test_temp_dir,
            test_conn,
            test_vectorstore,
            mock_embeddings,
            batch_size=10
        )
        
        # Verify document was inserted
        cursor = test_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count_1 = cursor.fetchone()[0]
        assert count_1 == 1, f"Expected 1 document after first ingestion, got {count_1}"
        
        # Verify doc_id is an integer
        assert isinstance(doc_id_1, int), f"Expected doc_id to be int, got {type(doc_id_1)}"
        
        # Second ingestion (duplicate)
        doc_id_2 = ingest_document(
            test_content,
            test_filename,
            test_temp_dir,
            test_conn,
            test_vectorstore,
            mock_embeddings,
            batch_size=10
        )
        
        # Verify same doc_id returned (duplicate detected)
        assert doc_id_1 == doc_id_2, f"Expected same doc_id for duplicate, got {doc_id_1} and {doc_id_2}"
        
        # Verify still only one row
        cursor.execute("SELECT COUNT(*) FROM documents")
        count_2 = cursor.fetchone()[0]
        assert count_2 == 1, f"Expected 1 document after duplicate ingestion, got {count_2}"
        
        # Verify content_hash is set correctly
        cursor.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id_1,))
        content_hash = cursor.fetchone()[0]
        expected_hash = compute_content_hash(test_content)
        assert content_hash == expected_hash, f"Content hash mismatch: got {content_hash[:16]}..., expected {expected_hash[:16]}..."
        
        print("✅ Regression test passed: Duplicate ingestion handled correctly")
        print(f"   - First ingestion: doc_id={doc_id_1}")
        print(f"   - Second ingestion: doc_id={doc_id_2} (duplicate detected)")
        print(f"   - Total documents: {count_2}")
        print(f"   - Content hash: {content_hash[:16]}...")
        return True
        
    except Exception as e:
        print(f"❌ Regression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        test_conn.close()
        shutil.rmtree(test_temp_dir, ignore_errors=True)

def test_schema_migration():
    """Test that schema migration works: create DB without content_hash, run ingestion, verify column is added."""
    import tempfile
    import shutil
    
    test_temp_dir = tempfile.mkdtemp(prefix="test_schema_migration_")
    
    try:
        # Create database WITHOUT content_hash column (legacy schema)
        test_conn = sqlite3.connect(":memory:", check_same_thread=False)
        test_conn.execute("""
            CREATE TABLE documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                mime TEXT,
                size_bytes INTEGER,
                created_at TEXT
            )
        """)
        test_conn.execute("""
            CREATE TABLE pages (
                doc_id INTEGER,
                page_number INTEGER,
                content_type TEXT,
                text TEXT,
                PRIMARY KEY (doc_id, page_number, content_type)
            )
        """)
        test_conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id INTEGER,
                file_name TEXT,
                page_start INTEGER,
                page_end INTEGER,
                chunk_index INTEGER,
                text TEXT
            )
        """)
        
        # Verify content_hash does NOT exist initially
        columns_before = get_table_columns(test_conn, "documents")
        assert "content_hash" not in columns_before, "content_hash should not exist initially"
        print(f"✅ Initial schema verified: columns={columns_before}")
        
        # Run ensure_schema to add missing column
        ensure_schema(test_conn)
        
        # Verify content_hash NOW exists
        columns_after = get_table_columns(test_conn, "documents")
        assert "content_hash" in columns_after, f"content_hash should exist after migration. Columns: {columns_after}"
        print(f"✅ Schema migration verified: columns={columns_after}")
        
        # Verify index exists
        cursor = test_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_documents_content_hash'")
        index_exists = cursor.fetchone() is not None
        assert index_exists, "UNIQUE index on content_hash should exist"
        print("✅ UNIQUE index verified")
        
        # Now test ingestion works with migrated schema
        test_vectorstore = init_chroma_db(test_temp_dir)
        test_content = b"Test document for schema migration"
        test_filename = "test.pdf"
        
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        
        mock_embeddings = MockEmbeddings()
        
        # Run ingestion (should work with migrated schema)
        doc_id = ingest_document(
            test_content,
            test_filename,
            test_temp_dir,
            test_conn,
            test_vectorstore,
            mock_embeddings,
            batch_size=10
        )
        
        # Verify document was inserted with content_hash
        cursor.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,))
        content_hash = cursor.fetchone()[0]
        expected_hash = compute_content_hash(test_content)
        assert content_hash == expected_hash, f"Content hash mismatch after migration"
        
        print(f"✅ Ingestion after migration successful: doc_id={doc_id}, hash={content_hash[:16]}...")
        print("✅ Schema migration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Schema migration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        test_conn.close()
        shutil.rmtree(test_temp_dir, ignore_errors=True)

def test_semantic_retrieval():
    """Test that semantic retrieval works: query term appears only in document #3 and model retrieves it."""
    import tempfile
    import shutil
    
    test_temp_dir = tempfile.mkdtemp(prefix="test_semantic_retrieval_")
    
    try:
        # Initialize database
        test_conn = init_sqlite_db()
        
        # Create 3 test documents with unique terms
        test_docs = [
            ("document1.pdf", b"Document 1 content about apples and oranges."),
            ("document2.pdf", b"Document 2 content about bananas and grapes."),
            ("document3.pdf", b"Document 3 content about unique_term_xyz123 and strawberries."),
        ]
        
        class MockEmbeddings:
            def __init__(self):
                self.model = "mock-embedding"
            
            def embed_documents(self, texts):
                # Simple mock: return different embeddings for different texts
                # Make embeddings for doc3 more distinct
                embeddings = []
                for i, text in enumerate(texts):
                    if "unique_term_xyz123" in text.decode('utf-8', errors='ignore').lower():
                        # Document 3 gets distinct embedding
                        embeddings.append([0.5] * 384)
                    else:
                        # Documents 1 and 2 get similar embeddings
                        embeddings.append([0.1 + (i % 2) * 0.01] * 384)
                return embeddings
            
            def embed_query(self, text):
                # Mock query embedding - make it similar to document 3
                if "unique_term_xyz123" in text.lower():
                    return [0.5] * 384  # Higher similarity to doc 3
                return [0.1] * 384
        
        mock_embeddings = MockEmbeddings()
        
        # Initialize vectorstore with embedding function
        test_vectorstore = init_chroma_db(test_temp_dir, mock_embeddings)
        
        # Ingest all 3 documents
        doc_ids = []
        for filename, content in test_docs:
            doc_id = ingest_document(
                content,
                filename,
                test_temp_dir,
                test_conn,
                test_vectorstore,
                mock_embeddings,
                batch_size=10
            )
            doc_ids.append(doc_id)
            print(f"Ingested {filename} → doc_id={doc_id}")
        
        # Create a chat model mock
        class MockChatModel:
            def invoke(self, prompt):
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                # Check if unique_term_xyz123 is in the prompt (context)
                if "unique_term_xyz123" in prompt:
                    return MockResponse("The unique term unique_term_xyz123 was found in document3.pdf")
                return MockResponse("NOT FOUND")
        
        mock_chat = MockChatModel()
        
        # Query for the unique term that only appears in document 3
        query = "unique_term_xyz123"
        print(f"\nQuerying for: {query}")
        print(f"Searching across ALL documents (no filter)")
        
        answer, sources = query_documents(
            query,
            test_vectorstore,
            mock_chat,
            ["All"],  # Search all files - should retrieve from doc3
            5,  # top_k
            test_conn,
            "mock-embedding",
            "mock-chat"
        )
        
        # Verify results
        print(f"\nAnswer: {answer}")
        print(f"Retrieved {len(sources)} sources")
        print(f"Retrieved file names: {[s.get('file_name') for s in sources]}")
        
        # CRITICAL: Check that document3.pdf was retrieved (not just document1.pdf)
        doc3_retrieved = any(
            source.get('file_name') == 'document3.pdf' 
            for source in sources
        )
        
        # Fail if only document1.pdf was retrieved (single-document search behavior)
        doc1_only = all(
            source.get('file_name') == 'document1.pdf' 
            for source in sources
        ) and len(sources) > 0
        
        assert not doc1_only, f"FAIL: Only document1.pdf retrieved (single-document search). Expected document3.pdf. Got: {[s.get('file_name') for s in sources]}"
        assert doc3_retrieved, f"FAIL: Expected document3.pdf to be retrieved, but got: {[s.get('file_name') for s in sources]}"
        
        # Verify we're retrieving across multiple documents (multi-document search)
        unique_files = set(s.get('file_name') for s in sources)
        print(f"   - Unique files in results: {unique_files}")
        
        # Check that answer mentions the unique term or document3
        assert "unique_term_xyz123" in answer.lower() or "document3" in answer.lower(), \
            f"Answer should mention unique_term_xyz123 or document3, got: {answer}"
        
        # Verify similarity scores are present
        for source in sources:
            assert 'similarity_score' in source, "Each source should have similarity_score"
            assert source['similarity_score'] >= 0.0, "Similarity score should be non-negative"
        
        # Verify top-k retrieval (at least 5 chunks)
        assert len(sources) >= 5, f"Expected at least 5 chunks, got {len(sources)}"
        
        print("✅ Semantic retrieval test passed!")
        print(f"   - Query: {query}")
        print(f"   - Retrieved {len(sources)} chunks across {len(unique_files)} document(s)")
        print(f"   - Retrieved documents: {[s.get('file_name') for s in sources]}")
        print(f"   - Similarity scores: {[s.get('similarity_score', 0.0) for s in sources]}")
        print(f"   - Multi-document search confirmed: {len(unique_files) > 1 if len(sources) > 1 else 'single result'}")
        return True
        
    except Exception as e:
        print(f"❌ Semantic retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        test_conn.close()
        shutil.rmtree(test_temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Run regression tests if requested via command line argument
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            print("Running duplicate ingestion test...")
            test_duplicate_ingestion()
        elif sys.argv[1] == "--test-migration":
            print("Running schema migration test...")
            test_schema_migration()
        elif sys.argv[1] == "--test-retrieval":
            print("Running semantic retrieval test...")
            test_semantic_retrieval()
        elif sys.argv[1] == "--test-all":
            print("Running all tests...")
            print("\n=== Test 1: Duplicate Ingestion ===")
            test_duplicate_ingestion()
            print("\n=== Test 2: Schema Migration ===")
            test_schema_migration()
            print("\n=== Test 3: Semantic Retrieval ===")
            test_semantic_retrieval()
    else:
        main()
