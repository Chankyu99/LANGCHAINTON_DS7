import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

class ItemRegulation(BaseModel):
    item_name: str = Field(description="물품의 이름 (예: 보조배터리, 신라면, 김치, 감기약)")
    item_keywords: List[str] = Field(description="Self-Querying 시 검색 정확도를 높이기 위한 물품 관련 핵심 키워드 리스트 (예: ['보조배터리', '배터리', '리튬', 'power bank'])")
    category: str = Field(description="물품 카테고리 (예: 전자기기, 농축산물/식품, 의약품, 액체류, 일반물품, 무기류, 인화성물질)")
    regulation_type: Literal["보안검색(국내선)", "보안검색(국제선)", "세관통관(미국)", "공통"] = Field(description="이 규정이 보안검색 관련인지 세관 통관 관련인지")
    
    # 기내/위탁 반입 여부 (주로 한국 출발지 보안검색용)
    carry_on: Literal["허용", "금지", "조건부 허용", "해당없음"] = Field(description="기내 반입 여부")
    checked_baggage: Literal["허용", "금지", "조건부 허용", "해당없음"] = Field(description="위탁 수하물 반입 여부")
    
    # 미국 입국 허용 여부 (미국 CBP 세관용)
    us_customs_admissibility: Literal["허용", "금지", "조건부 허용", "해당없음"] = Field(description="미국 입국 시 세관 통과 허용 여부 (주로 USDA, FDA 등 방역 통관 규정 반영)")
    
    reason_and_condition: str = Field(description="반입 금지 사유 또는 조건부 허용 시의 상세 조건 (100ml 이하, 100Wh 이하, 육류 성분 제외 등)")
    source: str = Field(description="데이터 출처 (한국 교통안전공단(avsec365), 인천공항, 미국 TSA, 미국 CBP, 미국 USDA, 미국 FDA 등)")

def build_transformer_chain(model_name: str = "gpt-5-mini"):
    """
    비구조화된 텍스트 청크를 받아 ItemRegulation 스키마에 맞는 정규화된 문서로 변환하는 LLM 체인을 생성합니다.
    """
    # 1. LLM 초기화 (gpt-5-mini 사용)
    # 실제 환경에서는 API 키가 환경 변수 OPENAI_API_KEY 로 설정되어 있어야 합니다.
    # Note: gpt-5-mini는 가칭이므로, 실제 가용한 최신 모델(gpt-4o-mini 등)로 대체될 수 있습니다.
    llm = ChatOpenAI(model=model_name, temperature=1)
    
    # 2. Structured Output 활성화 (Pydantic 스키마 주입)
    # 이 기능 덕분에 LLM의 응답은 항상 ItemRegulation 클래스의 JSON 구조를 띄게 됩니다.
    structured_llm = llm.with_structured_output(ItemRegulation)

    # 3. 데이터 추출 및 번역을 위한 프롬프트 정의
    prompt = PromptTemplate.from_template(
        """
        당신은 항공 보안 및 세관 규정 데이터 전처리 전문가입니다.
        주어진 원시 데이터(Raw Data) 텍스트를 분석하여, 제시된 통합 스키마 형식에 맞게 정보를 추출하세요.
        
        [지시사항]
        - 영어로 된 TSA나 CBP 규정인 경우, 물품명(item_name)과 상세 조건(reason_and_condition)은 반드시 "한국어"로 번역하여 작성하세요.
        - 특히 미국 입국 규정의 경우 한국인 방문객에게 중요한 "미국 농무부(USDA)의 농축산물(육류 등) 반입 제한"과 "식약처(FDA)의 의약품 통관 규정" 정보를 중점적으로 파악하여 us_customs_admissibility 및 이유에 반영하세요.
        - 데이터 출처(source) 판별: 한국어 텍스트라면 '인천공항' 또는 'avsec365'로, 영어 텍스트이자 세관/방역이면 '미국 CBP/USDA/FDA' 등으로 규정 성격에 맞게 기입하세요.
        - 키워드(item_keywords)는 해당 물품을 검색할 때 사용자들이 흔히 입력할 만한 유의어, 영어 원문, 줄임말 등을 포함하여 3~5개 정도 도출하세요.
        - Literal 타입으로 제한된 필드(예: carry_on)는 반드시 주어진 선택지("허용", "금지", "조건부 허용", "해당없음") 중에서만 선택하세요.
        
        [원시 데이터]
        {raw_text}
        """
    )

    # 4. 체인 연결: 프롬프트 -> 구조화된 LLM
    chain = prompt | structured_llm
    return chain

def process_chunk_to_document(raw_text: str, chain) -> Document:
    """
    원시 텍스트 청크를 체인에 통과시켜 LangChain Document 객체로 변환합니다.
    (page_content와 metadata 분리)
    """
    # 1. LLM을 통해 정규화된 JSON(Pydantic 객체) 추출
    structured_data: ItemRegulation = chain.invoke({"raw_text": raw_text})
    
    # 2. Vector DB 의미 검색(Vector Search) 대상이 될 본문 생성
    # 이름과 상세 규정을 합쳐서 문맥을 풍부하게 만듭니다.
    page_content = f"항목: {structured_data.item_name}\n규정 상세: {structured_data.reason_and_condition}"
    
    # 3. 메타데이터 필터링(Self-Querying 등)을 위한 속성 딕셔너리 생성
    metadata = {
        "item_name": structured_data.item_name,
        "item_keywords": structured_data.item_keywords,
        "category": structured_data.category,
        "regulation_type": structured_data.regulation_type,
        "carry_on": structured_data.carry_on,
        "checked_baggage": structured_data.checked_baggage,
        "us_customs_admissibility": structured_data.us_customs_admissibility,
        "source": structured_data.source
    }
    
    # 4. LangChain Document 리턴
    return Document(page_content=page_content, metadata=metadata)
