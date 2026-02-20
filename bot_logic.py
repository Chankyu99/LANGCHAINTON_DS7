"""
bot_logic.py
------------
schema.md 2~4단계 RAG 파이프라인 구현.

  2단계: Router & Slot Filling  — 대화에서 {출발지, 도착지, 물품} 추출
  3단계: Rewriter & Retriever   — 용어 정규화 + 메타데이터 필터 벡터 검색
  4단계: Judge & Generator      — 판정(🟢/🟡/🔴) + Bullet Point 답변 생성

단독 테스트:
    .venv/bin/python bot_logic.py
"""

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── 경로 / 상수 ────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = BASE_DIR / "chroma_db"
COLLECTION_NAME = "airline_regulations"
TOP_K           = 5          # 검색 결과 수
SCORE_THRESHOLD = 1.2        # 거리 임계값 (초과 시 Fallback)

# ── 모델 초기화 ────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

# ─────────────────────────────────────────────────────────────
# 2단계: 슬롯 추출 (Router & Slot Filling)
# ─────────────────────────────────────────────────────────────

SLOT_SYSTEM_PROMPT = """당신은 항공 규정 챗봇의 슬롯 추출기입니다.
사용자 메시지와 현재 대화 맥락에서 다음 4가지 슬롯을 JSON으로 추출하세요.

출력 형식 (반드시 순수 JSON만 출력):
{
  "departure": "출발 국가 코드 (KR/US/JP 등, 모르면 null)",
  "arrival": "도착 국가 코드 (KR/US/JP 등, 모르면 null)",
  "item": "물품명 (모르면 null)",
  "quantity": "수량/용량 등 속성 (모르면 null)"
}

규칙:
- 한국/대한민국 → KR, 미국 → US, 일본 → JP
- 국가 코드를 모르거나 언급이 없으면 null
- 국가가 같은 출발지/목적지인 경우도 그대로 추출"""


def extract_slots(user_message: str, chat_history: list[dict], current_slots: dict) -> dict:
    """
    대화 메시지에서 슬롯(출발지, 도착지, 물품, 속성)을 추출.
    기존 current_slots에 새로운 정보를 업데이트하여 반환.
    """
    history_text = ""
    for msg in chat_history[-6:]:  # 최근 6턴만 참조
        role = "사용자" if msg["role"] == "user" else "봇"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""현재 슬롯 상태: {json.dumps(current_slots, ensure_ascii=False)}

최근 대화:
{history_text}
사용자 최신 메시지: {user_message}

위 정보를 바탕으로 슬롯을 추출하세요. 기존에 확정된 슬롯은 유지하세요."""

    response = llm.invoke([
        SystemMessage(content=SLOT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        new_slots = json.loads(response.content.strip())
        # 기존 슬롯 유지 + 새 정보 병합
        merged = {**current_slots}
        for k, v in new_slots.items():
            if v is not None:
                merged[k] = v
        return merged
    except json.JSONDecodeError:
        return current_slots


def check_missing_slots(slots: dict) -> Optional[str]:
    """미확정 슬롯에 대한 재질문 문자열 반환. 모두 확정이면 None."""
    if not slots.get("departure") or not slots.get("arrival"):
        return "✈️ 어디에서 출발하여 어디로 가시나요? (예: 한국 → 미국)"
    if not slots.get("item"):
        return "🎒 어떤 물건의 반입 규정이 궁금하신가요?"
    if slots.get("departure") == slots.get("arrival"):
        return "⚠️ 출발지와 도착지가 같습니다. 다시 입력해 주세요."
    return None


# ─────────────────────────────────────────────────────────────
# 3단계: 쿼리 재작성 + 하이브리드 검색
# ─────────────────────────────────────────────────────────────

REWRITE_SYSTEM_PROMPT = """당신은 항공 보안 규정 전문가입니다.
사용자가 말한 물품명을 공식 항공 규정 용어로 변환하세요.
출력 형식 (순수 JSON):
{"canonical": "공식 용어", "synonyms": ["동의어1", "동의어2"]}

예시:
- 고추장 → {"canonical": "액체·분무·겔류", "synonyms": ["장류", "소스류"]}
- 보조배터리 → {"canonical": "리튬이온 배터리(보조배터리)", "synonyms": ["충전기", "파워뱅크"]}"""


def normalize_item(item: str) -> str:
    """물품명을 공식 규정 용어로 정규화하여 검색 쿼리 강화."""
    try:
        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"물품: {item}"),
        ])
        result = json.loads(response.content.strip())
        canonical = result.get("canonical", item)
        synonyms  = result.get("synonyms", [])
        return f"{item} {canonical} {' '.join(synonyms)}"
    except Exception:
        return item


def retrieve_docs(slots: dict) -> list[dict]:
    """
    확정된 슬롯으로 ChromaDB에서 관련 문서 검색.
    출발국(KR)과 도착국(US) 규정을 각각 검색하여 합산.
    """
    item      = slots.get("item", "")
    departure = slots.get("departure", "KR")
    arrival   = slots.get("arrival", "US")

    query = normalize_item(item)
    jurisdictions = list({departure, arrival})  # 중복 제거

    all_docs = []
    seen_ids = set()

    for jurisdiction in jurisdictions:
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=TOP_K,
            filter={"jurisdiction": jurisdiction},
        )
        for doc, score in results:
            doc_id = doc.metadata.get("doc_id", id(doc))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append({"doc": doc, "score": score, "jurisdiction": jurisdiction})

    return all_docs


# ─────────────────────────────────────────────────────────────
# 4단계: 최종 판정 + 답변 생성
# ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """당신은 항공 규정 챗봇 '기내뭐돼'입니다.
검색된 규정 문서를 바탕으로 사용자 질문에 답변하세요.

답변 규칙:
1. 첫 줄: 이모지 판정 결과
   - 🟢 반입 가능
   - 🟡 조건부 가능 (조건 명시 필요)
   - 🔴 반입 불가
2. 출/도착국 규정 중 단 하나라도 prohibited이면 → 🔴로 판정
3. 판정 이유를 2~3줄 Bullet Point(-)로 간결하게 정리
4. 검색된 규정 출처(stage, doc_type)를 한 줄로 명시
5. 불확실한 경우 억측하지 말고 정중히 안내 후 항공사 고객센터 연락 권고
6. 한국어로 친절하게 답변"""

FALLBACK_MSG = (
    "😓 죄송합니다. 해당 물품에 대한 규정 정보를 데이터베이스에서 찾지 못했습니다.\n\n"
    "정확한 정보를 위해 이용하실 **항공사 고객센터** 또는 "
    "**[항공보안365](https://www.avsec365.or.kr)**를 통해 확인해 주세요."
)


def generate_answer(user_message: str, slots: dict, retrieved: list[dict]) -> str:
    """검색 결과 기반 최종 답변 생성."""
    if not retrieved:
        return FALLBACK_MSG

    # 점수 임계값 초과 시 Fallback
    if all(r["score"] > SCORE_THRESHOLD for r in retrieved):
        return FALLBACK_MSG

    # 컨텍스트 구성
    context_parts = []
    for r in retrieved:
        doc  = r["doc"]
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('jurisdiction', '?')} 규정 / {meta.get('stage', '?')}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    departure = slots.get("departure", "?")
    arrival   = slots.get("arrival", "?")
    item      = slots.get("item", "?")

    prompt = f"""노선: {departure} → {arrival}
물품: {item}
사용자 질문: {user_message}

검색된 규정:
{context}

위 정보를 바탕으로 답변해주세요."""

    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content


# ─────────────────────────────────────────────────────────────
# 전체 파이프라인 진입점
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    user_message: str,
    chat_history: list[dict],
    slots: dict,
) -> tuple[str, dict]:
    """
    RAG 파이프라인 실행.

    Returns:
        (bot_response, updated_slots)
    """
    # 포괄적 질문 감지 (간단 키워드 기반)
    broad_keywords = ["다 알려", "전부", "모두", "목록", "리스트"]
    if any(kw in user_message for kw in broad_keywords) and not slots.get("item"):
        return (
            "🗂️ 어떤 카테고리의 규정이 궁금하신가요?\n\n"
            "아래 중 하나를 선택하거나, 직접 물품명을 입력해 주세요.\n"
            "- 🔫 총기·무기류\n"
            "- 💊 의약품·의료기기\n"
            "- 🧴 액체·겔·분무류\n"
            "- 🔋 배터리·전자기기\n"
            "- 🍎 식품·음식류\n"
            "- 💰 현금·귀중품",
            slots,
        )

    # 2단계: 슬롯 추출
    updated_slots = extract_slots(user_message, chat_history, slots)

    # 슬롯 미확정 시 재질문
    missing_q = check_missing_slots(updated_slots)
    if missing_q:
        return missing_q, updated_slots

    # 3단계: 검색
    retrieved = retrieve_docs(updated_slots)

    # 4단계: 답변 생성
    answer = generate_answer(user_message, updated_slots, retrieved)

    return answer, updated_slots


# ─────────────────────────────────────────────────────────────
# 단독 테스트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🛫 기내뭐돼 — 파이프라인 단독 테스트")
    print("=" * 60)

    test_cases = [
        {
            "desc": "노선+물품 모두 명시 (고추장, KR→US)",
            "message": "한국에서 미국 갈 때 고추장 기내 반입 가능해?",
            "slots": {},
        },
        {
            "desc": "노선 미확정 테스트",
            "message": "보조배터리 가져가도 돼?",
            "slots": {},
        },
        {
            "desc": "슬롯 이어받기 테스트",
            "message": "고추장이요",
            "slots": {"departure": "KR", "arrival": "US"},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {tc['desc']}")
        print(f"  입력: {tc['message']}")
        print(f"  슬롯: {tc['slots']}")
        response, new_slots = run_pipeline(tc["message"], [], tc["slots"])
        print(f"  → 업데이트 슬롯: {new_slots}")
        print(f"  → 응답:\n{response}")
        print("-" * 60)
