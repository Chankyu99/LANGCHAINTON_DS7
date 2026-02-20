from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 기존 모듈 재사용
from rag.retriever import get_query_analyzer_chain, build_retriever_with_filters
from rag.chain import get_rag_chain

def load_vector_db():
    """
    데이터베이스(ChromaDB) 객체를 생성하고 연결하는 함수입니다.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # etl 파이프라인에서 생성한 chroma_db 폴더 연동
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="regulations_kb")
    return db

def load_ai_models():
    """
    질문 분석 모델과 답변 생성 모델을 초기화합니다.
    """
    query_analyzer = get_query_analyzer_chain()
    qa_chain = get_rag_chain()
    return query_analyzer, qa_chain

def analyze_intent(prompt: str, query_analyzer):
    """
    사용자의 질문 의도(목적지, 카테고리 등)를 분석합니다.
    """
    return query_analyzer.invoke({"query": prompt})

def retrieve_documents(prompt: str, intent, db):
    """
    분석된 의도를 바탕으로 필터를 걸어 DB에서 정확한 문서를 찾아옵니다.
    """
    filtered_retriever = build_retriever_with_filters(db, intent)
    return filtered_retriever.invoke(prompt)

def generate_answer_stream(prompt: str, docs: list, qa_chain):
    """
    찾아온 문서 내용을 바탕으로 스트리밍 답변(조각생성)을 시작합니다.
    """
    context_text = "\n\n".join([f"[출처: {d.metadata.get('source', '알수없음')}]\n{d.page_content}" for d in docs])
    return qa_chain.stream({"context": context_text, "question": prompt})
