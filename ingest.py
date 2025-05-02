import os
import numpy as np 
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from DataIngestion import vector_data_ingestion


# Data Chunking
def chunkdata(page_text, title, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(page_text)

    chunked_output = []
    for i, chunk in enumerate(chunks):
        chunked_output.append({
            "title": title,
            "chunk_id": i + 1,
            "text": chunk
        })

    return chunked_output

def JsonChunking(file_paths):  # file_paths will have a list of all json files 
    with open(file_paths, "r", encoding="utf-8") as f:
        json_content = json.load(f)
        
    page_text_raw = json_content.get("content", {}).get("page_content", "")
    page_text = page_text_raw.replace("\n", "  ").replace('\\n', ' ')
    title = json_content.get("title", "")
    
    chunks = chunkdata(page_text, title)
    return chunks

"""# Data Embedding
def embedding_create(chunked_data):
    sentences = [chunk["text"] for chunk in chunked_data]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    embeddings = model.encode(sentences)
    
    # Check if embeddings are in the right format
    print("Embeddings shape:", embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings))  # Check dimensions
    print("Type of embeddings:", type(embeddings))  # Ensure it's a numpy array or list

    # Ensure embeddings are returned as a list or array
    return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings"""


#################### Main logic ##########################
# Getting file path
directory_path = r'C:\Users\KIIT\Desktop\GithubProj\Research\Extracted\extracted_json'
file_paths = []
for root, dirs, files in os.walk(directory_path):
    for file in files:
        file_paths.append(os.path.join(root, file))

i = 0
# Data Preprocessing  
  
for f in file_paths:
    i = i + 1
    ################ Chunking of each file ##################
    chunked_data = JsonChunking(f)
 
 
    ################ embedding  &  Ingesting #################
    index = f"index_{i}"
    checkT,result=vector_data_ingestion(chunked_data,index)

    
    if checkT == 1:
        print(f"Embedding and ingestion for {f} is successful")
        print("\nMy result:", result)
    else:
        print(f"Ingestion Failed for {f}")
