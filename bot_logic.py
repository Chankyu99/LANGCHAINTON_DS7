import os
import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db

# ==========================================
# 1. Pydantic Model (Entity Extraction용 스키마)
# ==========================================
class TravelInfo(BaseModel):
    departure: str = Field(description="Departure country code (e.g., 'KR' for South Korea, 'US' for United States). Empty string if not mentioned.")
    destination: str = Field(description="Destination country code (e.g., 'KR', 'US'). Empty string if not mentioned.")
    item: str = Field(description="The specific item the user wants to carry. Empty string if not mentioned.")

# ==========================================
# 2. 정보 추출 및 검증 모듈
# ==========================================
def extract_travel_info(user_question, llm):
    """
    사용자 질문에서 출발지, 도착지, 물품명을 구조화된 데이터(TravelInfo)로 추출합니다.
    """
    extraction_prompt = ChatPromptTemplate.from_template(
    "You are an aviation/customs expert. "
    "Extract the departure country, destination country, and the specific item from the question.\n"
    "- departure: ISO country code (e.g., KR, US)\n"
    "- destination: ISO country code (e.g., KR, US)\n"
    "- item: the product name\n\n"
    "Question: {item}"  # 변수명을 item으로 설정
    )
    
    # LangChain의 구조화 출력 기능 사용
    structured_llm = llm.with_structured_output(TravelInfo)
    
    # 위에서 {item}이라고 썼으니 여기서도 item=... 으로 넣어줍니다.
    return structured_llm.invoke(extraction_prompt.format(item=user_question))

def expand_query(item, llm):
    """
    물품명을 분석하여 세관 규정에 걸릴만한 숨겨진 성분과 카테고리로 확장합니다.
    """
    expansion_prompt = ChatPromptTemplate.from_template(
        "You are an expert aviation security and customs consultant. "
        "Your goal is to identify hidden ingredients or categories that might be restricted by customs (e.g., meat-based powders, processed meats, seeds, liquids, batteries).\n\n"
        "### Examples:\n"
        "- Item: '라면'\n  Keywords: 가공식품, 육류 추출 성분, 분말 스프, 유제품 성분\n"
        "- Item: '스팸'\n  Keywords: 가공육, 돼지고기 성분, 통조림, 육류 제품\n"
        "- Item: '보조배터리'\n  Keywords: 리튬 이온 배터리, 전자제품, 화기 엄금\n"
        "- Item: '미숫가루'\n  Keywords: 곡물류, 분말 제품, 가공식품\n\n"
        "### Task:\n"
        "Analyze the following item and list up to 5 keywords for customs retrieval. "
        "Focus on 'hidden' restricted ingredients that are not obvious from the name.\n"
        "Return ONLY the keywords in KOREAN, separated by commas.\n\n"
        "Item: {item}"
    )
    
    chain = expansion_prompt | llm | StrOutputParser()
    keywords = chain.invoke({"item": item})
    
    # 원본 품목명과 확장된 키워드를 합쳐서 검색 쿼리로 사용
    return f"{item}, {keywords}"

# ==========================================
# 3. 다이내믹 DB 검색 및 포맷팅 모듈
# ==========================================
def retrieve_docs(query, departure, destination, vectorstore):
    """출발지와 도착지 변수를 동적으로 받아 해당 국가의 규정만 필터링합니다."""
    dep_docs = vectorstore.similarity_search(query=query, k=2, filter={"jurisdiction": departure})
    dest_docs = vectorstore.similarity_search(query=query, k=2, filter={"jurisdiction": destination})
    return dep_docs + dest_docs

def format_docs_with_metadata(docs, departure, destination):
    formatted_context = ""
    for doc in docs:
        meta = doc.metadata
        jurisdiction = meta.get('jurisdiction', '')
        item_name = meta.get('item', 'Unknown')
        
        # 동적으로 출발지/도착지 라벨링
        if jurisdiction == departure:
            formatted_context += (
                f"[{jurisdiction} Departure Rules]\n"
                f"- Item: {item_name}\n"
                f"- Cabin Baggage: {meta.get('cabin_decision', 'N/A')}\n"
                f"- Checked Baggage: {meta.get('checked_decision', 'N/A')}\n"
                f"- Details: {doc.page_content}\n\n"
            )
        elif jurisdiction == destination:
            formatted_context += (
                f"[{jurisdiction} Arrival Rules]\n"
                f"- Item: {item_name}\n"
                f"- Entry Decision: {meta.get('decision', 'N/A')}\n"
                f"- Details: {doc.page_content}\n\n"
            )
    return formatted_context

# ==========================================
# 4. 메인 실행 함수
# ==========================================
def ask_to_ai(user_question, current_info):
    llm = ChatOpenAI(model="gpt-5-mini")
    vectorstore = get_vectorstore()
    
    # 1단계: 현재 질문에서 새로운 정보 추출
    new_info = extract_travel_info(user_question, llm)
    
    # 2단계: 정보 병합 (챗봇이 기억)
    updated_info = {
        "departure": new_info.departure if new_info.departure else current_info.get("departure", ""),
        "destination": new_info.destination if new_info.destination else current_info.get("destination", ""),
        "item": new_info.item if new_info.item else current_info.get("item", "")
    }
    
    # 병합된 최종 정보 객체 생성
    info = TravelInfo(**updated_info)
    
    # 3단계: 필수 정보 검증
    if not info.departure or not info.destination:
        return "어디에서 출발하여 어디로 가시나요? 노선 정보를 알려주세요. (예: 한국에서 미국으로)", [], updated_info
    
    if info.departure == info.destination:
        return "출발지와 도착지가 같습니다. 노선 정보를 다시 한 번 확인해주세요.", [], updated_info
        
    if not info.item:
        return "어떤 물건의 반입 규정이 궁금하신가요? (예: 라면, 보조배터리)", [], updated_info
        
    # 4단계: 쿼리 확장 및 검색
    expanded_query = expand_query(info.item, llm)
    retrieved_docs = retrieve_docs(expanded_query, info.departure, info.destination, vectorstore)
    
    # 검색 결과가 없을 때도 리턴값 3개를 맞춰줘야 합니다.
    if not retrieved_docs:
        return f"데이터베이스에서 '{info.item}'에 대한 규정 정보를 찾을 수 없습니다.", [], updated_info
    
    # 5단계: 프롬프트 주입용 메타데이터 포맷팅
    context = format_docs_with_metadata(retrieved_docs, info.departure, info.destination)
    
    # 6단계: 최종 답변 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert aviation security and customs assistant.
        The user is traveling from {departure} to {destination}.
        Answer the user's question about the item '{item}' based strictly on the provided [Context] below.
        
        Organize your response clearly with the following sections:
        1. Cabin Baggage (Departure from {departure})
        2. Checked Baggage (Departure from {departure})
        3. Customs & Entry (Arrival in {destination})
        
        Handling Missing Items:
        If the exact item is NOT explicitly mentioned in the [Context], DO NOT say "Information not available" (확인 불가). 
        Instead, explain that the item is not explicitly listed in the restricted/prohibited database, meaning it is likely permitted but subject to general baggage and customs guidelines (e.g., liquids, agricultural products).
        
        Please respond in Korean language naturally and politely.
        
        [Context]
        {context}
        """),
        ("human", "Question: {question}\nExpanded Keywords: {expanded_query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "departure": info.departure,
        "destination": info.destination,
        "item": info.item,
        "context": context, 
        "question": user_question,
        "expanded_query": expanded_query
    })
    
    # 마지막 리턴도 3개
    return answer, retrieved_docs, updated_info