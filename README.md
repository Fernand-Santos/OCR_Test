# Document RAG with Ollama

A Streamlit application that ingests PDFs and images, extracts text, chunks content, stores embeddings in ChromaDB, and enables question-answering using Ollama models.

## Features

- **Ollama Server Detection**: Real-time status checking and connection monitoring
- **Model Management**: View and select from downloaded Ollama models
- **Drag & Drop Upload**: Visual file upload area with progress tracking
- **Document Ingestion**: Upload PDFs and images for processing
- **Text Extraction**: PDF text extraction via PyMuPDF, OCR hooks for images
- **Chunking**: Intelligent text chunking using LangChain
- **Vector Storage**: ChromaDB for efficient similarity search
- **RAG Querying**: Ask questions with context-aware answers using Ollama models
- **File Filtering**: Query specific documents or all documents
- **SQLite Inspection**: Ephemeral database viewer for transparency
- **Document Management**: Clean deletion of documents from all storage
- **Status Dashboard**: Overview metrics for documents, models, and server status

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running locally
   - Download from: https://ollama.ai/
   - Start Ollama service

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull required Ollama models:**
   ```bash
   # Embedding model (recommended)
   ollama pull nomic-embed-text

   # Chat model (choose one)
   ollama pull qwen2.5
   # or
   ollama pull ministral
   ```

## Usage

1. **Start the application:**
   ```bash
   streamlit run OCR.py
   ```

2. **Configure Ollama:**
   - Check server status in the sidebar
   - Click "Refresh Models" to load available models
   - View downloaded models in expandable sections
   - Select embedding and chat models from dropdowns

3. **Upload Documents:**
   - Use the drag-and-drop area to upload PDFs or images
   - See file count and sizes before processing
   - Click "Process Documents" to ingest and process files
   - Monitor progress with real-time status updates
   - Documents are stored with SHA256-based IDs

4. **Ask Questions:**
   - Enter questions in the text area (supports multi-line)
   - Optionally filter by specific files using the sidebar
   - Click "Search & Answer" for AI-powered responses
   - View answers with source citations and page references

5. **Inspect Storage:**
   - Use the "SQLite Storage Viewer" expander to examine:
     - Document metadata
     - Page-level text extraction
     - Text chunks with metadata
     - Retrieval logs
     - Last retrieval mapping

6. **Manage Documents:**
   - View stored documents list with delete buttons
   - Delete documents (removes from all storage locations)

## Architecture

### Components
- **Streamlit UI**: Web interface for all interactions
- **Runtime Storage**:
  - Temporary directory for raw files
  - Ephemeral SQLite database (in-memory)
- **Processing Pipeline**:
  - PDF extraction via PyMuPDF (Fitz)
  - OCR integration stub (DeepSeek-ready)
  - LangChain text chunking
- **RAG System**:
  - Ollama embeddings for vectorization
  - ChromaDB vector storage
  - Ollama chat models for answering
  - LangChain retrieval chains

### Data Flow
1. **Ingestion**: File → SHA256 ID → Text Extraction → Chunking → Embedding → ChromaDB + SQLite
2. **Querying**: Question → Embedding → Retrieval (with filters) → Context → Chat Model → Answer + Citations
3. **Deletion**: Document ID → Remove from SQLite + ChromaDB + Temp Files

## Key Features

- **Ollama-Only**: All embeddings and chat use Ollama models
- **Ephemeral SQLite**: Runtime-only database for inspection/debugging
- **Scoped Retrieval**: RAG results filtered to selected files when specified
- **Clean Deletion**: Complete removal from all storage locations
- **Model Flexibility**: Easy switching between Ollama models

## Troubleshooting

### Ollama Issues
- Ensure Ollama is running: `ollama serve`
- Check base URL in sidebar (default: http://localhost:11434)
- Verify models are pulled: `ollama list`

### Import Errors
- Install dependencies: `pip install -r requirements.txt`
- Ensure Python 3.8+ is being used

### Performance
- Large documents may take time to process
- Consider adjusting chunk size/overlap in code for optimization
- Monitor ChromaDB persistence directory disk usage

## Future Enhancements

- DeepSeek OCR API integration for images
- Batch processing capabilities
- Advanced chunking strategies
- Conversation history
- Export functionality for answers/sources