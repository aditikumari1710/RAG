

research-rag-app/
│
├── data/                        # PDF research papers
├── ingest.py                    # Load, chunk, embed & call DataIngestion.py for ingestion
├──DataIngestion.py              #ChromaDB  for storing embedding
├──verify_retrivl.py             #checking if data is ingested
├── gemini_test_llm.py              # RetrievalQA 
├── app.py                       # streamlit
├── requirements.txt             #all the requirement file
