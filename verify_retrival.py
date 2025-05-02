###for testing purpose-check if vector is stored or not ########

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Define the same embedding model used in ingestion
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 2: Load the Chroma DB from disk
vectordb = Chroma(
    collection_name="my_collection",  # Same name as ingestion
    persist_directory="./chroma_db",  # Same directory
    embedding_function=embedder,
)

# Step 3: Perform a similarity search
query = "Deep Residual Learning for Image Recognition"
retrieved_docs = vectordb.similarity_search(query, k=4)

# Step 4: Print retrieved documents
print(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
for doc in retrieved_docs:
    print("Page content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("---")
