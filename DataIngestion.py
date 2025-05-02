import os, json
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def vector_data_ingestion(chunked_data, index_id, persist_dir="./chroma_db", batch_size=500):
    """
    Ingests chunked data into Chroma vector store with batching.
    """
    try:
        # 1) Build Document objects
        docs = []
        for chunk in chunked_data:
            docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "title":    chunk["title"],
                    "chunk_id": chunk["chunk_id"]
                }
            ))

        # 2) Create embedding model (must match your query-time model!)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # 3) Instantiate Chroma and add documents in batches
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="my_collection"  
        )
        
        total_docs = len(docs)
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i+batch_size]
            vector_store.add_documents(
                documents=batch,
                ids=[f"{index_id}_{j}" for j in range(i, min(i + batch_size, total_docs))]
            )
            logger.info(f"Ingested batch {i//batch_size + 1} of {total_docs // batch_size + 1}")

        # 4) Persist to disk
        vector_store.persist()
        count = vector_store._collection.count()
        logger.info(f" Ingested {total_docs} chunks → stored {count} embeddings in `{persist_dir}`")

        return True, vector_store

    except Exception as e:
        logger.error(f"[ERROR] Ingestion failed: {e}")
        return False, None
