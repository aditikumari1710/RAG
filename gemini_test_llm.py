
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import configparser
import uuid

# Load the config file
config = configparser.ConfigParser()
config.read('config.properties')
GEMINI_API_KEY=config['API_KEY']['gemini.api']
gemini_model=config['Model']['gemini']
emb_model=config['Model']['embedding']
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(model_name=gemini_model)
embedding_model = HuggingFaceEmbeddings(model_name=emb_model)

vectordb = Chroma(
        persist_directory="./chroma_db",
        collection_name="my_collection",
        embedding_function=embedding_model
    )


def call_llm_context(context, query,source='None'):
    prompt = f"""
You are an intelligent, friendly AI assistant designed to give clear, informative, and conversational answers.

Your behavior is guided by the following rules:
1. If relevant context is available, use it to provide a detailed and helpful answer (minimum 100 tokens).
2. Do NOT hallucinate — if the context is irrelevant or not helpful for the question, respond with: "No context found."
3. If the user is just casually chatting (e.g., greetings or simple conversation), respond in a natural, friendly tone — like a smart chatbot.
4. Do not repeat or copy-paste context. Instead, explain or summarize clearly.
5. Explain it like you are a professor of AI/ML.
6. At the end of your answer, if a variable called 'source' has a value other than "None", append a new line: Source: {source}

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = gemini_model.generate_content(
    prompt,
    generation_config={
        "temperature": 0.2,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 600
    }
    )

        return response.text.strip()
    except Exception as e:
        print("Error calling Gemini model:", e)
        return "Error generating response."




def get_title(query):
    results = vectordb.similarity_search(query, k=3)
    metadatas=[]
    for i, doc in enumerate(results, 1):
        metadatas.append(doc.metadata.get('title'))
        
    return metadatas[0]
    
    
    
    

def clean_context(context):
    seen = set()
    cleaned = []
    for line in context.split("\n"):
        if line.strip() and line not in seen:
            cleaned.append(line)
            seen.add(line.strip())
    return "\n".join(cleaned)


def rag_pipeline(query,source):
    docs = vectordb.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    context = clean_context(context)

    answer = call_llm_context(context, query,source)
    return answer

def user_uploading_doc():
    pass

if __name__ == "__main__":
    try:
        user_query = input("Chatbot: Hey, enter topic to get context:\n")
        source_name=get_title(user_query)
        llm_response = rag_pipeline(user_query,source_name)
        print("The bot response: \n",llm_response)
    except Exception as e:
        print("The Exception occured :\n",e)
    
   