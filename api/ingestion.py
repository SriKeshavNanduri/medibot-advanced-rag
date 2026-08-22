from asyncio.log import logger

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import FastEmbedSparse

from langchain_qdrant import QdrantVectorStore, RetrievalMode

import os
import pickle
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

def main():
    load_dotenv("creds.env")
    my_gemini_key = os.getenv("GOOGLE_API_KEY")
    # 1. Verify Gemini API credentials exist
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable is missing.")
        print("Please run: export GOOGLE_API_KEY='your_key' (Linux/Mac) or set GOOGLE_API_KEY='your_key' (Windows CMD)")
        return

    print("Initializing embedding pipelines...")
    
    # Dense embeddings — semantic understanding using Gemini API
    dense_embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL"),  # Targets the latest Gemini 2 text embedding space,
        google_api_key = my_gemini_key
    )

    # Sparse embeddings — structural token matching
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", batch_size=256)

    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    file_path = os.getenv("CHUNKS_FILE_PATH")

    # 2. Extract serialized document chunks
    try:
        print(f" Loading document chunks from '{file_path}'...")
        with open(file_path, "rb") as file:  # 'rb' means read binary
            pdf_docs = pickle.load(file)
        
        print(f" Data successfully loaded! Extracted {len(pdf_docs)} structural chunks.")
        
    except FileNotFoundError:
        print(f" Error: The file at {file_path} was not found. Place it in this exact folder.")
        return
    except Exception as e:
        print(f" An error occurred while loading the pickle file: {e}")
        return

    # 3. Stream data pipeline to your native local network port 
    LOCAL_QDRANT_URL = os.getenv("LOCAL_QDRANT_URL")
    # print(f" Connecting to local standalone Qdrant at {LOCAL_QDRANT_URL}...")
    # print(f" Encoding chunks & building index for collection '{COLLECTION_NAME}'... (This may take a moment)")

    try:
        # Initiates automated schema config, batch embedding computation, and network upsertion.
        vectorstore = QdrantVectorStore.from_documents(
            documents=pdf_docs,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url=LOCAL_QDRANT_URL,      
            collection_name=COLLECTION_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        print(" Ingestion Complete! Data successfully vectorized and upserted into Qdrant.")
        
    except Exception as e:
        print(f"Critical error during Qdrant ingestion: {e}")
        print("Make sure your background qdrant.exe terminal window is actively open and running.")


def ingest_pdfs(pdf_docs, skipped_chunks = 0):  
    load_dotenv("creds.env")
    my_gemini_key = os.getenv("GOOGLE_API_KEY")
    # 1. Verify Gemini API credentials exist
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable is missing.")
        print("Please run: export GOOGLE_API_KEY='your_key' (Linux/Mac) or set GOOGLE_API_KEY='your_key' (Windows CMD)")
        return

    print("Initializing embedding pipelines...")
    
    # Dense embeddings — semantic understanding using Gemini API
    dense_embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL"),  # Targets the latest Gemini 2 text embedding space,
        google_api_key = my_gemini_key
    )

    # Sparse embeddings — structural token matching
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", batch_size=256)

    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    # 2. Extract serialized document chunks
    try: 
        
        # 3. Stream data pipeline to your native local network port 
        LOCAL_QDRANT_URL = os.getenv("LOCAL_QDRANT_URL")
        # print(f" Connecting to local standalone Qdrant at {LOCAL_QDRANT_URL}...")
        # print(f" Encoding chunks & building index for collection '{COLLECTION_NAME}'... (This may take a moment)")

        # Initiates automated schema config, batch embedding computation, and network upsertion.
        vectorstore = QdrantVectorStore(
            client=LOCAL_QDRANT_URL,
            documents=pdf_docs,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url=LOCAL_QDRANT_URL,      
            collection_name=COLLECTION_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        vectorstore.add_documents(pdf_docs)


        logger.info(" Ingestion Complete! Data successfully vectorized and upserted into Qdrant.")

        return {
            "status": "successfully_ingested",
            "num_chunks": len(pdf_docs),
            "docs": pdf_docs,
            "skipped_chunks": skipped_chunks,
        }
        
    except Exception as e:
            
        return {
            "status": "failed",
            "num_chunks": 0,
            "docs": [],
            "skipped_chunks": 0,
            "error": str(e),
        }




# This acts as the script entrypoint to execute your ingestion routine
if __name__ == "__main__":
    main()
