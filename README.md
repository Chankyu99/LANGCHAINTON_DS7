# ✈️ 기내뭐돼 (What Can I Bring?)

> 모두의연구소 데이터사이언티스트 7기 랭체인톤  
> **RAG 기반 국가/항공사별 수하물 반입 규정 AI 챗봇**

---

## 서비스 소개

**기내뭐돼**는 공식 법령 및 항공 보안 규정 원문 데이터를 바탕으로, 사용자가 원하는 물품의 기내/위탁 반입 가능 여부를 대화형으로 안내하는 RAG 챗봇입니다.

- 🟢 **반입 가능** / 🟡 **조건부 가능** / 🔴 **반입 불가** 로 직관적인 판정 제공
- 출발국·도착국 규정을 **교차 검색**하여 가장 엄격한 기준으로 판정
- 슬롯 필링 방식으로 **노선 → 물품 → 속성** 순서로 대화 흐름 관리

---

## 프로젝트 구조

```
LANGCHAINTON_DS7/
├── app.py              # Streamlit 챗봇 UI
├── bot_logic.py        # RAG 파이프라인 핵심 로직 (슬롯필링, 검색, 판정, 생성)
├── ingest.py           # 데이터 임베딩 스크립트 (최초 1회 실행)
├── requirements.txt    # 의존성 패키지 목록
│
├── data/
│   └── index_docstore_export.jsonl   # 항공 규정 원문 데이터 (84개 문서)
│
├── RAG_pipeline/
│   └── schema.md       # RAG 파이프라인 설계 계획서
│
├── docs/               # 추가 문서 (추후 작성)
└── prompt/             # 프롬프트 모음 (추후 작성)
```

---

## 빠른 시작

### 1. 가상환경 세팅

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성 후 OpenAI API 키 입력:

```
OPENAI_API_KEY=sk-...
```

### 3. 데이터 임베딩 (최초 1회)

```bash
.venv/bin/python ingest.py
```

### 4. 앱 실행

```bash
.venv/bin/streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## RAG 파이프라인 4단계

| 단계 | 이름 | 내용 |
|------|------|------|
| 1단계 | Data Ingestion | JSONL → text-embedding-3-small → ChromaDB |
| 2단계 | Router & Slot Filling | GPT로 노선·물품 슬롯 추출, 미확정 시 재질문 |
| 3단계 | Rewriter & Retriever | 용어 정규화 + 메타데이터 필터 벡터 검색 |
| 4단계 | Judge & Generator | 이모지 판정 + Bullet Point 답변 생성 |

자세한 설계는 [RAG_pipeline/schema.md](./RAG_pipeline/schema.md) 참고.

---

## 기술 스택

- **LLM**: GPT-4o-mini (OpenAI)
- **Embedding**: text-embedding-3-small (OpenAI)
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **UI**: Streamlit
