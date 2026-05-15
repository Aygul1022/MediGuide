import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("1. Reading clinical data...")

docs = []
limit = int(os.getenv("VECTOR_DB_LIMIT", "0"))  # 0 means index the full dataset.

with open("data/clinical_qa_processed.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if limit and i >= limit:
            break
        data = json.loads(line)
        # Yapay zekanın okuyacağı metni İngilizce kurguluyoruz
        question = data.get('question', '').replace('Answer this question truthfully ', '').strip()
        answer = data.get('answer', '').strip()
        text = f"Clinical case summary:\nSymptoms/Presentation: {question}\nFindings/Outcome: {answer}"
        
        docs.append(Document(page_content=text, metadata={"source": data.get("source", "Unknown")}))

print(f"2. Converting {len(docs)} cases into vector embeddings...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

print("✅ SUCCESS! Vector Database has been successfully created.")
