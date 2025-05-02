from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import configparser

# Load the config file
config = configparser.ConfigParser()
config.read('config.properties')

HF_TOKEN=config['API_KEY']['hugging.face']

#HF_TOKEN = "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"  # use your actual token here
MODEL_ENDPOINT="https://api-inference.huggingface.co/models/google/flan-t5-base"
# The model you want to use (ensure the model endpoint is correct)
def call_llm(context, query):
    prompt = f"""
You are an AI assistant trained to provide meaningful and detailed answers based on the given context. Please use the context provided below to directly answer the user's question, focusing on key insights and explanations without repeating any text. Your answer should be unique, informative.Atleast token size should be 100 
Context:
{context}

Question: {query}

Answer:
"""
    print("\nThe context passed\n \n",context)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,  # Increased for more detailed responses
            "min_new_tokens": 50,
            "temperature": 0
        }
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
    #print("\nLLM response: \t", response)
    
    if response.status_code == 200:
        try:
            response_json = response.json()
            print("DEBUG - Response JSON:", response_json)
            #if the response is a list and has a generated_text key return the generated_text
            if isinstance(response_json, list) and "generated_text" in response_json[0]:
                return response_json[0]["generated_text"]
            #if the response is a dict and has a generated_text key return the generated_text   
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




def clean_context(context):
    # Remove exact duplicate sentences or paragraphs
    seen = set()
    cleaned = []
    for line in context.split("\n"):
        if line.strip() and line not in seen:
            cleaned.append(line)
            seen.add(line.strip())
    return "\n".join(cleaned)





def rag_pipeline(query):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(
    persist_directory="./chroma_db",
    collection_name="my_collection",
    embedding_function=embedding_model)

    # This directly gets the top-k documents related to the query
    docs = vectordb.similarity_search(query, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])
    context = clean_context(context)

    answer = call_llm(context, query)
    return answer


user_query = input("Chatbot:Hey, Enter topic to get context: \n")
llm_response = rag_pipeline(user_query)
print("\nLLM response: \n", llm_response)
