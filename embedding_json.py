"""
[Data Embedding Script]
목적: JSONL 형식의 항공 보안/세관 규정 데이터를 임베딩하여 Chroma DB에 저장합니다.
임베딩 모델: text-embedding-3-small
사용법: 
1. 하단의 'TARGET_FILE_NAME' 변수에 임베딩할 원본 파일명을 지정합니다. (원본 데이터 폴더: ./data/)
2. 터미널에서 'python embedding_json.py'를 실행합니다.
"""

# ==========================================
# ⚙️ 실행 설정 (Configuration)
# ==========================================
TARGET_FILE_NAME = "kor-jfk.jsonl"


import os
import json
import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Streamlit secrets에서 API 키를 가져와 환경변수에 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def ingest_data(file_name):
    # 파일 경로 설정 (data 폴더 안에 있다고 가정)
    file_path = f"./data/{file_name}"
    
    print(f"1. {file_name} 파일 읽기 시작...")
    docs = []
    
    # 파일을 한 줄씩 읽어서 LangChain Document 객체로 변환
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("page_content", "")
            metadata = data.get("recommended_metadata", {})
            
            doc = Document(page_content=text, metadata=metadata)
            docs.append(doc)
            
    print(f"총 {len(docs)}개의 문서를 읽었습니다.")
    
    # 기존에 존재하는 chroma_db에 임베딩하여 추가(Append)
    print("2. 임베딩 및 Chroma DB 저장 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("ChromaDB에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    ingest_data(TARGET_FILE_NAME)