# import statements

from tracemalloc import start

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_core.documents import Document
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling.chunking import HybridChunker



from docling.datamodel.settings import settings 
settings.inference.compile_torch_models = False


import os
import time
import pickle # Keep this for saving the final docs
from dataclasses import dataclass, field

import hashlib

import psycopg2
from psycopg2.extras import DictCursor
from api.ingestion import ingest_pdfs



# =====================================
# load environment variables
load_dotenv("creds.env")

# ==========================================
# FUNCTIONS 

# functions for metadata extraction

def get_chunk_type(chunk) -> str:
    """Maps Docling item labels to: 'text', 'table', 'heading', or 'code'."""
    if not chunk.meta or not chunk.meta.doc_items:
        return "text"
    
    # Extract raw label strings/enums from all items in this chunk
    labels = [str(item.label).lower() for item in chunk.meta.doc_items]
    
    # Priority mapping logic
    if any("table" in l for l in labels):
        return "table"
    elif any("code" in l for l in labels):
        return "code"
    elif any("section_header" in l or "title" in l or "heading" in l for l in labels):
        return "heading"
    else:
        return "text"

def get_section_title(chunk) -> str:
    """Extracts the immediate or full heading context."""
    if chunk.meta and chunk.meta.headings:
        # Returns immediate parent section heading: e.g., '2. Patient Confidentiality'
        return chunk.meta.headings[-1] 
        
        # Alternatively, for full breadcrumbs (e.g., 'Section 1 > 2. Patient Confidentiality'):
        # return " > ".join(chunk.meta.headings)
    return ""

def get_source_document(chunk) -> str:
    """Extracts the original filename from the DocChunk origin metadata."""
    if chunk.meta and chunk.meta.origin and chunk.meta.origin.filename:
        return chunk.meta.origin.filename
    return "open_source"

# The collection -> roles mapping now lives in api/rbac.py so that the ingestion
# path and the FastAPI service cannot drift apart. api.rbac has no heavy imports.
from api.rbac import get_access_based_on_collection  # noqa: E402,F401

def get_directory_name(file_path: str) -> str:
    return os.path.basename(os.path.dirname(file_path))


# ======================================================================================


my_postgres_pwd = os.getenv("POSTGRES_PWD")
db_name = os.getenv("DB_NAME")
import urllib.parse
safe_password = urllib.parse.quote_plus(str(my_postgres_pwd))
db_user = os.getenv("DB_USER")


def ensure_chunker_initialized() -> None:
    """Initialize the shared Docling converter/chunker once using the same logic as the batch ingestion flow."""
    global _converter, _chunker

    _converter = DocumentConverter()
    local_bpe_tokenizer = tiktoken.get_encoding("cl100k_base")
    docling_tokenizer = OpenAITokenizer(tokenizer=local_bpe_tokenizer, max_tokens=384)
    _chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        max_tokens=384,
        merge_peers=True,
    )


def _process_single_file(file_path: str, folder_name: str) -> dict:
    """
    Runs in a worker process. Must only return picklable objects
    (plain dicts/lists/strings) — not the converter/chunker themselves.
    """
    global _converter, _chunker
 

    start = time.perf_counter()
    conn = None
    try:
        docling_doc = _converter.convert(file_path).document
        chunk_iter = _chunker.chunk(dl_doc=docling_doc)

        pdf_docs = []
        skipped_chunks = 0
        
        # Connect to default PostgreSQL on port 5432
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database=db_name,
            user=db_user,
            password=my_postgres_pwd  # Replace with your actual password
        )
        db_cursor = conn.cursor(cursor_factory=DictCursor)

        for chunk in chunk_iter:
            chunk_text = _chunker.contextualize(chunk=chunk)
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()

            # PostgreSQL syntax for checking existence
            db_cursor.execute("SELECT 1 FROM chunk_history WHERE chunk_hash = %s", (chunk_hash,))
            if db_cursor.fetchone():
                skipped_chunks += 1
                continue  # Skip this chunk as it's already ingested

            metadata = {
                "source_document": get_source_document(chunk),
                "collection": folder_name,
                # Was hardcoded to "general", which handed every chunk all five
                # roles and made the metadata.access_roles RBAC filter a no-op.
                "access_roles": get_access_based_on_collection(folder_name),
                "section_title": get_section_title(chunk),
                "chunk_type": get_chunk_type(chunk),
            }

            # PostgreSQL syntax uses %s placeholders instead of ?
            db_cursor.execute(
                """
                INSERT INTO chunk_history (chunk_hash, chunk_text, source_documentation, collection, chunk_type, section_title, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    chunk_hash,
                    chunk_text,
                    metadata["source_document"],
                    metadata["collection"],
                    metadata["chunk_type"],
                    metadata["section_title"],
                )
            )

            pdf_docs.append(Document(page_content=chunk_text, metadata=metadata))

        conn.commit()
        db_cursor.close()
        conn.close()
        if len(pdf_docs) == 0:
            return {
                "status": "no_new_chunks_to_ingest",
                "num_chunks": 0,
                "docs": [],
                "skipped_chunks": skipped_chunks,
            }

        
        ingestion_status = ingest_pdfs(pdf_docs, skipped_chunks=skipped_chunks)
        elapsed = time.perf_counter() - start

        ingestion_status["elapsed_seconds"] = elapsed

        return ingestion_status

    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        elapsed = time.perf_counter() - start
        ingestion_status["elapsed_seconds"] = elapsed
        return ingestion_status



# Renamed function to reflect sequential execution
def chunking_of_pdfs_sequential(folder_names: list[str], save_path: str | None = None):
    """
    Processes PDF files sequentially for chunking.
    max_workers defaults to os.cpu_count() if not specified.
    """
    current_dir = os.getcwd()
    all_file_paths = []
    for folder_name in folder_names:
        data_dir = os.path.join(current_dir, "mediassist_data", folder_name)
        if os.path.isdir(data_dir):
            files = os.listdir(data_dir)
            file_paths = [os.path.join(data_dir, f) for f in files]
            all_file_paths.extend(file_paths)
        else:
            print(f"Warning: Directory not found and will be skipped: {data_dir}")

    if not all_file_paths:
        print(f"No files found in the specified folders: {folder_names}")
        return []

    print(f"Starting sequential chunking for {len(all_file_paths)} files.")

    all_docs = []
    results_summary = []

    # Initialize converter and chunker once for sequential processing
    global _converter, _chunker
    _converter = DocumentConverter()
    local_bpe_tokenizer = tiktoken.get_encoding("cl100k_base")
    docling_tokenizer = OpenAITokenizer(tokenizer=local_bpe_tokenizer, max_tokens=384)
    _chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        max_tokens=384,
        merge_peers=True,
    )

    completed = 0
    for path in all_file_paths:
        completed += 1
        try:
            # Call the processing function directly
            result = _process_single_file(path, get_directory_name(path))
            results_summary.append(result)

            if result["status"] == "success":
                all_docs.extend(result["docs"])
                skipped_info = ""
                if result.get("skipped_chunks", 0) > 0:
                    skipped_info = f" (skipped {result['skipped_chunks']} existing chunks)"

                print(
                    f"[{completed}/{len(all_file_paths)}] OK  "
                    f"{os.path.basename(path)} -> {result['num_chunks']} chunks "
                    f"({result['elapsed_seconds']:.2f}s){skipped_info}"
                )

            else:
                print(f"[{completed}/{len(all_file_paths)}] FAIL {os.path.basename(path)}: {result['error']}")

        except Exception as e:
            print(f"[{completed}/{len(all_file_paths)}] ERROR {os.path.basename(path)}: {e}")
            results_summary.append({
                "file_path": path, "status": "error", "num_chunks": 0,
                "docs": [], "elapsed_seconds": None, "error": str(e),
            })

    # Summary
    succeeded = sum(1 for r in results_summary if r["status"] == "success")
    failed = len(results_summary) - succeeded
    print("---" * 30)
    print(f"Done. {succeeded} succeeded, {failed} failed. Total chunks: {len(all_docs)}")
    
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_docs, f)
        print(f"Saved {len(all_docs)} Document objects to {save_path}")

    return all_docs, results_summary
    
# Removed _init_worker as it's no longer needed for sequential processing
# The global _converter and _chunker are now initialized directly in chunking_of_pdfs_sequential
_converter = None
_chunker = None


if __name__ == "__main__":
    docs, summary = chunking_of_pdfs_sequential( # Call the sequential function
        ["general", "clinical", "nursing", "equipment", "billing"], # Example with multiple folders
        save_path="policies_chunks_all_files.pkl",
    )
