import os
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def build_vector_db(
    documents: List[Document],
    db_directory: str = "./chroma_db",
    collection_name: str = "regulations_kb"
) -> Chroma:
    """
    변환된 LangChain Document 리스트를 입력받아 Chroma DB에 적재합니다.
    (Self-Querying 및 한영 통합 검색을 위한 다국어 지원 임베딩 적용)
    
    Args:
        documents: etl/transformer.py를 거쳐 생성된 Document 객체 리스트
        db_directory: 로컬 테스트를 위해 저장될 Vector DB 디렉토리 경로
        collection_name: 적재될 컬렉션 이름
        
    Returns:
        적재가 완료된 Chroma VectorStore 인스턴스
    """
    # 1. 다국어 임베딩 설정
    # text-embedding-3-small 모델은 gpt-5-mini와 함께 사용하기에 적합하며,
    # 한글/영어 텍스트를 같은 벡터 공간에 매핑하여 언어에 구애받지 않는 검색을 가능하게 합니다.
    # 환경 변수 OPENAI_API_KEY 가 설정되어 있어야 합니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Chroma DB 생성 및 데이터 적재
    # 주어진 문서를 임베딩 모델로 벡터화 한 후, 지정된 경로에 영구 저장합니다.
    # 이미 해당 경로에 동일한 컬렉션이 있다면 기존 데이터에 추가됩니다.
    print(f"📦 총 {len(documents)} 개의 문서 벡터화를 시작합니다...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name
    )
    
    print(f"✅ Chroma DB 적재 완료! (Path: {db_directory})")
    
    # Chroma 인스턴스 리턴 (이후 RAG router 파이프라인의 retriever로 활용됨)
    return vectorstore

def get_retriever():
    """
    기존에 적재된 Chroma DB에서 벡터 검색기(Retriever)를 가져오는 헬퍼 함수입니다.
    이후 RAG 파이프라인에서 Metadata 필터나 Self-Querying을 적용할 때 활용됩니다.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="regulations_kb"
    )
    # k=3: 유사도 기준 상위 3개의 문서를 반환 (이후 파이프라인에서 search_kwargs로 필터 조건 추가 가능)
    return vectorstore.as_retriever(search_kwargs={"k": 3})
