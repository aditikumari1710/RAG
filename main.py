from fastapi import FastAPI
from typing import Union
import gemini_test_llm
from gemini_test_llm import rag_pipeline
from gemini_test_llm import get_title
import uuid

session_id=str(uuid.uuid4())
app=FastAPI()


#our root directory
@app.get("/")
def root():
    return {"enter query"}



#user will give there query here 
@app.post(f"/USER_query")
def get_query(Topic:str):
    session_id=str(uuid.uuid4())
    source=get_title(Topic)
    llm_response=rag_pipeline(Topic,source)
   
    return llm_response

