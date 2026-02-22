"""
bot_logic.py
------------
schema.md 2~4단계 RAG 파이프라인 구현.

  2단계: Router & Slot Filling  — 대화에서 {출발지, 도착지, 물품} 추출
  3단계: Rewriter & Retriever   — DB 목록 기반 항목 매핑 + 메타데이터 필터 벡터 검색
  4단계: Judge & Generator      — 판정(🟢/🟡/🔴) + Bullet Point 답변 생성

[v2 개선]
  - normalize_item() → map_item_to_db() 로 교체
    : LLM이 자유 생성하던 방식 → DB 84개 항목 목록 직접 참조 후 선택
    : "칼" → DB에서 "날 길이 6cm 초과 칼", "도끼·손도끼·큰 식칼 등 절단용 칼" 매핑
    : "미숫가루" → DB에서 US "가공/캔 식품" 카테고리 매핑

[v3 개선]
  - DB 매핑 전체 실패 시 단순 Fallback → LLM 일반 지식 기반 답변으로 격상
    : LLM이 물품을 추론 → 유사 카테고리 판단 → 항공 전문 지식으로 답변
    : 답변 하단에 ⚠️ "DB 미등재 항목, 일반 규정 기반 안내" 단서 명시
    : "보조배터리", "드라이기", "뜨개바늘" 등 DB 없는 물품 처리 가능

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
DATA_FILE       = BASE_DIR / "data" / "index_docstore_export.jsonl"
COLLECTION_NAME = "airline_regulations"
TOP_K           = 5          # 검색 결과 수
MAX_MAPPED      = 3          # LLM이 선택할 최대 DB 항목 수

# ── 모델 초기화 ────────────────────────────────────────────────
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
llm         = ChatOpenAI(model="gpt-5-mini", temperature=0)
advanced_llm = ChatOpenAI(model="gpt-5.2", temperature=0.2)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ─────────────────────────────────────────────────────────────
# DB 항목 목록 로드 (앱 시작 시 1회)
# ─────────────────────────────────────────────────────────────

def load_db_items() -> dict[str, list[str]]:
    """
    JSONL에서 국가별 item 목록을 로드.
    반환: {"KR": [...], "US": [...]}
    """
    items: dict[str, list[str]] = {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            country = rec.get("country", "?")
            item    = rec.get("item", "")
            if country not in items:
                items[country] = []
            if item and item not in items[country]:
                items[country].append(item)
    return items


# 모듈 로드 시 1회만 실행
DB_ITEMS: dict[str, list[str]] = load_db_items()


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
    """대화 메시지에서 슬롯(출발지, 도착지, 물품, 속성)을 추출."""
    history_text = ""
    for msg in chat_history[-6:]:
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
# 3단계: DB 목록 기반 항목 매핑 + 검색 (v2 핵심 개선)
# ─────────────────────────────────────────────────────────────

MAP_SYSTEM_PROMPT = """당신은 항공 규정 DB의 항목 매핑 전문가입니다.

사용자가 입력한 물품명이 아래 DB 항목 목록 중 어느 것과 관련이 있는지 판단하세요.

규칙:
1. 사용자 물품이 DB 항목에 **직접 포함**되거나 **상위 개념**이면 해당 항목 선택
   - 예: "칼" → "날 길이 6cm 초과 칼", "도끼·손도끼·큰 식칼 등 절단용 칼"
   - 예: "총" → "모든 종류의 총기(권총·라이플·엽총 등)"
2. 사용자 물품이 DB 항목 카테고리에 **속하는 하위 개념**이면 해당 항목 선택
   - 예: "미숫가루" → "가공/캔 식품", "농산물/식품"
   - 예: "선글라스" → 관련 항목 없음
3. 관련 항목은 최대 {max_mapped}개까지만 선택
4. 관련 항목이 **전혀 없으면** 빈 리스트 반환

출력 형식 (반드시 순수 JSON 배열):
["항목명1", "항목명2"]  또는  []"""


def map_item_to_db(item: str, jurisdictions: list[str]) -> dict[str, list[str]]:
    """
    [v2 핵심] 사용자 물품명 → DB 항목 목록에서 관련 항목 선택.

    각 jurisdiction(KR/US)별로 관련 DB 항목을 선택하여 반환.
    반환: {"KR": ["날 길이 6cm 초과 칼", ...], "US": [...]}
    """
    result: dict[str, list[str]] = {}

    for jur in jurisdictions:
        db_list = DB_ITEMS.get(jur, [])
        if not db_list:
            result[jur] = []
            continue

        db_list_str = "\n".join(f"  - {it}" for it in db_list)

        prompt = f"""사용자 물품: "{item}"

[{jur}] DB 항목 목록:
{db_list_str}

위 DB 항목 중, 사용자 물품 "{item}"과 관련된 항목을 골라주세요."""

        response = llm.invoke([
            SystemMessage(content=MAP_SYSTEM_PROMPT.format(max_mapped=MAX_MAPPED)),
            HumanMessage(content=prompt),
        ])

        try:
            # 코드블록 등 감싸진 경우 정리
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            mapped = json.loads(raw)
            # DB에 실제로 있는 항목만 필터
            valid = [m for m in mapped if m in db_list]
            result[jur] = valid[:MAX_MAPPED]
        except (json.JSONDecodeError, TypeError):
            result[jur] = []

    return result


def retrieve_docs(slots: dict) -> tuple[list[dict], bool]:
    """
    확정된 슬롯으로 ChromaDB에서 관련 문서 검색.
    [v2] DB 목록 매핑 결과로 검색 쿼리를 구성.
    [v3] 매핑 전체 실패 여부를 두 번째 반환값으로 제공.

    Returns:
        (retrieved_docs, all_mapping_failed)
        - all_mapping_failed=True  → 모든 jurisdiction에서 매핑 실패 → v3 일반지식 Fallback 필요
    """
    item          = slots.get("item", "")
    departure     = slots.get("departure", "KR")
    arrival       = slots.get("arrival", "US")
    jurisdictions = list({departure, arrival})

    # DB 항목 매핑
    mapped = map_item_to_db(item, jurisdictions)
    print(f"[retrieve_docs] mapped: {mapped}")  # 디버그 로그

    # 모든 jurisdiction에서 매핑이 비었는지 체크
    all_mapping_failed = all(len(mapped.get(jur, [])) == 0 for jur in jurisdictions)

    all_docs = []
    seen_ids = set()

    for jur in jurisdictions:
        matched_items = mapped.get(jur, [])

        if matched_items:
            query = " ".join(matched_items) + " " + item
        else:
            query = item

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=TOP_K,
            filter={"jurisdiction": jur},
        )

        for doc, score in results:
            doc_id       = doc.metadata.get("doc_id", id(doc))
            db_item_name = doc.metadata.get("item", "")

            if matched_items:
                if db_item_name in matched_items:
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append({
                            "doc": doc, "score": score,
                            "jurisdiction": jur, "mapped": True
                        })
            else:
                if score <= 1.2 and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append({
                        "doc": doc, "score": score,
                        "jurisdiction": jur, "mapped": False
                    })

    return all_docs, all_mapping_failed


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
4. 검색된 규정 출처(jurisdiction / stage)를 한 줄로 명시
5. 사용자 물품이 DB 항목의 하위 개념일 경우, 어떤 카테고리 규정을 참조했는지 자연스럽게 언급
   - 예: "'칼'은 '날 길이 6cm 초과 칼' 규정이 적용됩니다."
6. 불확실한 경우 억측하지 말고 정중히 안내 후 항공사 고객센터 연락 권고
7. 한국어로 친절하게 답변"""

# [v3] DB에 없는 물품용 일반 지식 기반 프롬프트
GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """당신은 항공 보안 및 세관 규정 전문가 챗봇 '기내뭐돼'입니다.
사용자가 물어본 물품이 규정 데이터베이스에 등록되지 않은 항목입니다.

다음 절차로 답변하세요:
1. 물품의 특성을 파악하여 어떤 항공 규정 카테고리에 해당할지 추론
   (예: 보조배터리 → 리튬이온 배터리류 → IATA 위험물 규정 적용)
2. 국제 항공 규정(IATA·TSA·ICAO) 일반 기준으로 판정
3. 노선(출발국·도착국)별로 다를 수 있는 규정 차이 언급
4. 답변 형식:
   - 첫 줄: 🟢/🟡/🔴 판정 + 한 줄 요약
   - Bullet Point 2~3줄로 조건·수치 안내
   - 마지막 줄: ⚠️ 단서 문구 (아래 고정 문구 사용)
5. 확신이 없는 부분은 억측하지 않고 "항공사 확인 권장" 명시
6. 한국어로 답변

고정 단서 문구 (반드시 마지막에 포함):
⚠️ 이 답변은 공식 DB에 없는 항목으로, IATA 일반 항공 규정 기반 안내입니다. 정확한 규정은 이용 항공사 또는 [항공보안365](https://www.avsec365.or.kr)에서 확인하세요."""

FALLBACK_MSG = (
    "😓 죄송합니다. 해당 물품에 대한 규정 정보를 데이터베이스에서 찾지 못했습니다.\n\n"
    "정확한 정보를 위해 이용하실 **항공사 고객센터** 또는 "
    "**[항공보안365](https://www.avsec365.or.kr)**를 통해 확인해 주세요."
)


def generate_answer(user_message: str, slots: dict, retrieved: list[dict]) -> str:
    """[DB 매핑 성공] 검색 결과 기반 최종 답변 생성."""
    context_parts = []
    for r in retrieved:
        doc  = r["doc"]
        meta = doc.metadata
        context_parts.append(
            f"[{meta.get('jurisdiction', '?')} 규정 / {meta.get('stage', '?')}]\n"
            f"항목: {meta.get('item', '?')}\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    departure = slots.get("departure", "?")
    arrival   = slots.get("arrival", "?")
    item      = slots.get("item", "?")

    prompt = f"""노선: {departure} → {arrival}
사용자가 물어본 물품: {item}
사용자 질문: {user_message}

검색된 규정:
{context}

위 규정을 바탕으로 답변해주세요.
만약 사용자 물품이 DB 항목의 하위 개념이라면(예: '칼' → '날 길이 6cm 초과 칼'), 어떤 규정을 참조했는지 자연스럽게 안내해주세요."""

    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return response.content


def general_knowledge_answer(user_message: str, slots: dict) -> str:
    """[v3] DB 매핑 전체 실패 시 LLM 일반 항공 지식으로 추론·답변."""
    departure = slots.get("departure", "?")
    arrival   = slots.get("arrival", "?")
    item      = slots.get("item", "?")

    prompt = f"""노선: {departure} → {arrival}
사용자가 물어본 물품: "{item}"
사용자 질문: {user_message}

이 물품은 규정 DB에 등록되지 않은 항목입니다.
물품의 특성을 추론하고, 국제 항공 규정(IATA·TSA) 일반 기준으로 답변해주세요."""

    response = advanced_llm.invoke([
        SystemMessage(content=GENERAL_KNOWLEDGE_SYSTEM_PROMPT),
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
    """RAG 파이프라인 실행. Returns: (bot_response, updated_slots)"""

    # 포괄적 질문 감지
    broad_keywords = ["다 알려", "전부", "모두", "목록", "리스트"]
    if any(kw in user_message for kw in broad_keywords) and not slots.get("item"):
        return (
            "🗂️ 어떤 카테고리의 규정이 궁금하신가요?\n\n"
            "아래 중 하나를 선택하거나, 직접 물품명을 입력해 주세요.\n"
            "- 🔫 총기·무기류\n"
            "- 🔪 도검·공구류\n"
            "- 💊 의약품·의료기기\n"
            "- 🧴 액체·겔·분무류\n"
            "- 🔋 배터리·전자기기\n"
            "- 🍎 식품·농산물\n"
            "- 💰 현금·귀중품",
            slots,
        )

    # 2단계: 슬롯 추출
    updated_slots = extract_slots(user_message, chat_history, slots)

    # 슬롯 미확정 시 재질문
    missing_q = check_missing_slots(updated_slots)
    if missing_q:
        return missing_q, updated_slots

    # 3단계: DB 매핑 + 검색
    retrieved, all_mapping_failed = retrieve_docs(updated_slots)

    # 4단계: 답변 생성
    if retrieved:
        # DB에서 문서를 찾은 경우 → 정규 RAG 답변
        answer = generate_answer(user_message, updated_slots, retrieved)
    elif all_mapping_failed:
        # [v3] DB 매핑 자체가 전혀 안 된 경우 → LLM 일반 지식 답변
        print(f"[run_pipeline] DB 매핑 실패 → 일반 지식 Fallback: {updated_slots.get('item')}")
        answer = general_knowledge_answer(user_message, updated_slots)
    else:
        # 매핑은 됐으나 score 초과로 문서 없음 → 기존 Fallback
        answer = FALLBACK_MSG

    return answer, updated_slots


# ─────────────────────────────────────────────────────────────
# 단독 테스트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🛫 기내뭐돼 v3 — 일반지식 Fallback 테스트")
    print("=" * 60)

    test_cases = [
        {
            "desc": "v3 신규: 보조배터리 (DB 미등재)",
            "message": "한국→미국 보조배터리 기내 반입 가능해?",
            "slots": {},
        },
        {
            "desc": "v3 신규: 드라이기 (DB 미등재)",
            "message": "드라이기는 가져갈 수 있어?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
        {
            "desc": "v2 유지: 칼 (DB 간접 매핑)",
            "message": "칼은?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
        {
            "desc": "v2 유지: 미숫가루 (US 식품 매핑)",
            "message": "미숫가루는?",
            "slots": {"departure": "KR", "arrival": "US"},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {tc['desc']}")
        print(f"  입력: {tc['message']}")
        response, new_slots = run_pipeline(tc["message"], [], tc["slots"])
        print(f"  → 슬롯: {new_slots}")
        print(f"  → 응답:\n{response}")
        print("-" * 60)
