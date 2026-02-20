from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 사용자의 질문에서 필터링 조건을 뽑아낼 Pydantic 스키마
class SearchIntent(BaseModel):
    item_name: str = Field(description="질문에서 언급된 핵심 물품명 (예: 김치, 보조배터리, 컵라면)")
    target_country: Optional[str] = Field(None, description="목적지 국가 (예: 미국, 일본 등) 언급이 없으면 None")
    transport_method: Optional[str] = Field(None, description="'기내', '위탁', '알수없음' 중 하나")

def get_query_analyzer_chain(model_name: str = "gpt-5-mini"):
    """
    사용자의 자연어 질문을 입력받아 SearchIntent 객체(구조화된 필터 정보)로 변환하는 체인
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(SearchIntent)
    
    prompt = PromptTemplate.from_template(
        """
        당신은 공항 검색대 및 세관 안내 요원입니다.
        사용자의 질문을 분석하여, 검색에 필요한 핵심 키워드와 조건을 추출하세요.
        
        [사용자 질문]
        {query}
        """
    )
    
    return prompt | structured_llm

def build_retriever_with_filters(vectorstore, intent: SearchIntent):
    """
    분석된 intent(의도)를 바탕으로 Chroma VectorDB의 retriever를 필터와 함께 생성합니다.
    (Phase 1에서 구축한 ItemRegulation 스키마의 메타데이터 활용)
    """
    # 기본 필터 조건 (MongoDB/Chroma 문법 적용)
    filter_conditions = []
    
    # 1. 미국행이 포함된 경우 세관통관 필터 및 금지물품 배제
    if intent.target_country == "미국":
        filter_conditions.append({"regulation_type": {"$in": ["세관통관(미국)", "공통", "보안검색(국제선)"]}})
        # 미국 입국 시 절대 안 되는 것들을 미리 걸러낼 수 있도록 metadata 필터링 가능
        # (예시: filter_conditions.append({"us_customs_admissibility": {"$ne": "금지"}}))
        
    # 2. 수하물 형태에 따른 명시적 필터
    if intent.transport_method == "기내":
        filter_conditions.append({"carry_on": {"$in": ["허용", "조건부 허용"]}})
    elif intent.transport_method == "위탁":
        filter_conditions.append({"checked_baggage": {"$in": ["허용", "조건부 허용"]}})

    # 결합된 필터가 있으면 적용, 없으면 기본 retriever
    search_kwargs = {"k": 3}
    if filter_conditions:
        if len(filter_conditions) > 1:
            search_kwargs["filter"] = {"$and": filter_conditions}
        else:
            search_kwargs["filter"] = filter_conditions[0]
            
    # 최종 필터 조건이 적용된(Hallucination 방지) 검색기 반환
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
