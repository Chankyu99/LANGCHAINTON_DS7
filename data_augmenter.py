"""
data_augmenter.py
-----------------
data/index_docstore_export.jsonl 파일을 불러와서, 
각 규정 항목(item) 및 원문 텍스트에 대해 사람들이 자주 검색할 만한 
구체적인 동의어/예시 품목을 GPT-4o-mini로 자동 생성(Augmentation)합니다. 

증강된 결과물은 data/index_docstore_augmented.jsonl로 저장됩니다.

실행:
    .venv/bin/python data_augmenter.py
"""

import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── 환경변수 로드 ──────────────────────────────────────────────
load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "index_docstore_export.jsonl"
OUTPUT_FILE= DATA_DIR / "index_docstore_augmented.jsonl"

# ── 모델 초기화 ────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

SYSTEM_PROMPT = """당신은 항공 보안 규정 데이터베이스 구축 전문가입니다.
사용자가 제공하는 '항공 규정 카테고리(항목)'와 '규정 원문'을 읽고,
실제 여행객들이 이 카테고리를 검색할 때 자주 입력할 만한 구체적인 물건 이름, 동의어, 하위 품목, 대표 브랜드명 등을 10개에서 15개 정도 유추해 주세요.

출력 규칙:
1. 오직 쉼표(,)로만 구분된 단어 리스트만 출력하세요. (예: 샴푸, 로션, 선크림, 바디워시)
2. 설명이나 부연 설명은 절대 덧붙이지 마세요.
3. 관련성이 매우 높은 단어만 선별하세요."""

def load_jsonl(filepath: Path) -> list[dict]:
    """JSONL 파일을 읽어 dict 리스트로 반환."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def augment_record(rec: dict) -> dict:
    """단일 레코드에 대해 연관 검색어를 생성하고 page_content에 덧붙임."""
    item         = rec.get("item", "")
    page_content = rec.get("page_content", "")
    
    prompt = f"규정 항목명: {item}\n규정 원문: {page_content}\n\n이 규정과 연관된 구체적인 검색 키워드 10~15개를 쉼표로 구분해서 출력해줘."
    
    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        keywords = response.content.strip()
        
        # 원본 내용 + 생성된 키워드를 하나의 텍스트로 합침
        augmented_content = f"{page_content}\n\n[연관 검색 키워드]: {keywords}"
        
        # 레코드 복사 후 업데이트
        new_rec = rec.copy()
        new_rec["page_content"] = augmented_content
        return new_rec
        
    except Exception as e:
        print(f"Error augmenting item '{item}': {e}")
        return rec  # 에러 발생 시 원본 그대로 반환

def main():
    print("=" * 60)
    print("🛫 기내뭐돼 — LLM 기반 데이터 증강(Augmentation) 시작")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"❌ 원본 데이터 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    records = load_jsonl(INPUT_FILE)
    print(f"📂 원본 레코드 {len(records)}개를 로드했습니다.")
    
    augmented_records = []
    
    print("\n⏳ GPT-4o-mini를 통해 각 문서에 동의어/검색어 증강 중...")
    # tqdm으로 진행률 표시
    for rec in tqdm(records, desc="데이터 증강"):
        aug_rec = augment_record(rec)
        augmented_records.append(aug_rec)
        # Rate limit 방지를 위한 짧은 대기 시간
        time.sleep(0.5)

    # ── 결과 저장 ──────────────────────────────────────────────
    print(f"\n💾 증강 완료! 결과를 저장합니다: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in augmented_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print("✅ 데이터 증강 파이프라인이 성공적으로 종료되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
