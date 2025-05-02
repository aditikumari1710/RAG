from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Hugging Face Token (make sure it's valid and has inference permissions)
HF_TOKEN = "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"  # use your actual token here

# The model you want to use
#MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#MODEL_ENDPOINT="https://api-inference.huggingface.co/models/bert-base-uncased"
#MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
#MODEL_ENDPOINT="https://api-inference.huggingface.co/models/google/flan-t5-base"
#MODEL_ENDPOINT="https://api-inference.huggingface.co/models/bert-base-uncased"
#MODEL_ENDPOINT = "https://api-inference.huggingface.co/pipeline/text2text-generation/google/flan-t5-small"
#512 ebedding size
MODEL_ENDPOINT=""
#flan-t5-base-768 embedding size
def call_llm(context, query):
    prompt = f"""You are a helpful AI assistant. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"""
    print("Context retrived \t:",context)
    """payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7
        }
    }"""
    payload={
        "inputs":"prompt"
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        MODEL_ENDPOINT,
        json=payload,
        headers=headers
    )
    print("\nLLM response: \t",response)
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("DEBUG - Response JSON:", response_json)
            if isinstance(response_json, list) and "generated_text" in response_json[0]:
                return response_json[0]["generated_text"]
            elif isinstance(response_json, dict) and "generated_text" in response_json:
                return response_json["generated_text"]
            else:
                return "Error: Unexpected response format."
        except ValueError as e:
            print("Error parsing response:", e)
            return "Error parsing response."
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)
        return "Error: Invalid response from the model."


# Load your embeddings & vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#768 VECTOR DIMENSION
vectordb = Chroma(
    persist_directory="./chroma_db3",
    collection_name="my_collection",
    embedding_function=embedding_model
)

# Set up retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


def rag_pipeline(query):
    docs = retriever.invoke(query) 
    print("\ndocs\n",docs)
    context = "\n\n".join([doc.page_content for doc in docs]) #getting context from vector db
    answer = call_llm(context, query)
    return answer
    
    # DEBUG: count & sample retrieval
collection = vectordb._collection
print("▶️ Total embeddings stored:", collection.count())

# try a hard‑coded test query
test_docs = vectordb.as_retriever(search_kwargs={"k":1}).invoke("machine learning")
print("▶️ Sample retrieval:", test_docs)

    


User_qury = input("Chatbot: What is your query? ")
llm_response = rag_pipeline(User_qury)
print("LLM:\t", llm_response)
# After you instantiate vectordb…
#collection = vectordb._collection  # the underlying Chroma collection
#print("Total embeddings stored:", collection.count())  
