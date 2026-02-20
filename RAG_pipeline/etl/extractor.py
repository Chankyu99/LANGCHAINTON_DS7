import os
from typing import List

# 임시 Document 생성을 위한 모듈 (향후 진짜 로더로 대체)
from langchain_core.documents import Document

def extract_raw_data(data_path: str = None) -> List[str]:
    """
    향후 실제 데이터 형식(PDF, CSV, DB 등)이 확정될 때 
    해당 문서 로더(Loader)와 Text Splitter를 주입할 인터페이스입니다.
    
    현재는 데이터 전처리 형태가 보류되었으므로, 테스트 파이프라인 작동을 위해
    임의의 한국어/영어 혼합 비구조화 텍스트 청크를 반환합니다.
    """
    print("⏳ [Extractor] 실제 데이터 형식 확정 전까지 하드코딩된 샘플 데이터를 반환합니다.")
    
    sample_chunks = [
        # 1. 한국어 보안검색 데이터 (avsec365 / 인천공항 기준: 액체류 / 김치 등)
        "국내선 비행기 이용 시 김치와 고추장은 기내 반입이 가능하지만, 국제선의 경우 액체류로 분류되어 기내 반입이 엄격히 금지됩니다. "
        "국제선 승객은 김치, 고추장 등의 액체 함유 식품을 반드시 위탁수하물로 처리해야 합니다.",
        
        # 2. 미국 CBP/USDA 영문 세관 데이터 (육류, 농축산물)
        "U.S. Customs and Border Protection (CBP) strictly prohibits the entry of most meat, poultry, and products containing meat into the United States. "
        "This is an enforcement of USDA regulations to prevent animal diseases. Items like instant noodles (ramen) containing meat extract in their seasoning packets are not admissible and must be declared.",
        
        # 3. 미국 FDA 영문 세관 데이터 (의약품)
        "When traveling to the United States with medications, adhering to FDA guidelines is required. "
        "Prescription drugs should be in their original containers with the doctor's prescription printed on the bottle. You may only bring up to a 90-day supply of FDA-approved medication."
    ]
    
    return sample_chunks
