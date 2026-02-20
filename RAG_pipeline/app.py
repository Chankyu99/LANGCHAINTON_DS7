import streamlit as st
import os

# 스트림릿 금고(.streamlit/secrets.toml)에서 키를 꺼내서 환경변수에 세팅합니다.
# 이렇게 하면 LangChain 내부의 OpenAI 관련 모듈들이 알아서 이 키를 사용하게 됩니다.
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# AI 핵심 두뇌 역할을 하는 로직 파일만 가져옵니다. (UI 코드 0줄)
import bot_logic

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="한-미 통합 규정 AI 비서", page_icon="✈️", layout="centered")

st.title("✈️ 똑똑한 공항/세관 AI 비서")
st.markdown("""
이 챗봇은 **한국의 공공데이터(표 형식)**와 **미국 CBP/TSA 규정(Q&A 형식)**이  
하나의 통일된 스키마로 정규화(ETL)된 RAG 파이프라인 위에서 동작합니다.
""")

# --- 세션 초기화 (대화 기록 보관) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DB 및 모델 캐싱 ---
# @st.cache_resource 장식자를 사용하면, 스트림릿이 화면을 새로고침할 때마다 무거운 DB나 체인을 다시 로드하지 않고 메모리에서 꺼내옵니다.
@st.cache_resource 
def init_rag_system():
    # 1. 벡터 DB 로드
    db = bot_logic.load_vector_db()
    # 2. 분석/답변 모델 로드
    query_analyzer, qa_chain = bot_logic.load_ai_models()
    return db, query_analyzer, qa_chain

try:
    vectorstore, query_analyzer, qa_chain = init_rag_system()
except Exception as e:
    st.error("RAG 파이프라인 초기화 중 에러가 발생했습니다. DB 경로 및 Secrets 설정을 확인해주세요.")
    st.stop()

# --- UI: 기존 채팅 기록 화면에 뿌리기 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 메인 채팅 로직 ---
if prompt := st.chat_input("예: 컵라면(육류스프 내장) 미국 갈 때 위탁으로 부쳐도 되나요?"):
    
    # 1. 사용자 질문 화면 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 봇 답변 생성 영역
    with st.chat_message("assistant"):
        with st.status("🔍 사용자의 의도를 분석하고 규정을 검색 중입니다...", expanded=True) as status:
            
            st.write("1️⃣ 질문 의도 파악 중...")
            # UI는 모르는 복잡한 분석을 bot_logic에게 외주줍니다.
            intent = bot_logic.analyze_intent(prompt, query_analyzer)
            st.write(f"📝 *파악된 의도: 목적지='{intent.target_country}', 수하물형태='{intent.transport_method}', 핵심키워드='{intent.item_name}'*")
            
            st.write("2️⃣ 필터가 결합된 단일 스키마 문서 검색 중...")
            # 문서를 찾아오는 심부름도 bot_logic이 대신합니다.
            docs = bot_logic.retrieve_documents(prompt, intent, vectorstore)
            
            if not docs:
                status.update(label="⚠️ 관련된 규정을 찾지 못했습니다.", state="error")
                st.stop()
                
            st.write("3️⃣ 발견된 규정(한/미 혼합):")
            for i, d in enumerate(docs, 1):
                st.caption(
                    f"💡 문서 {i}: {d.metadata.get('category')} | 제한: 기내({d.metadata.get('carry_on')}), "
                    f"위탁({d.metadata.get('checked_baggage')}), 미국입국({d.metadata.get('us_customs_admissibility')})"
                )
            status.update(label="✅ 검색 완료! 답변을 생성합니다.", state="complete", expanded=False)

        # 3. 완성된 답변을 스트리밍(타이핑 효과)으로 화면에 뿌려줍니다.
        message_placeholder = st.empty()
        full_response = ""
        
        # 실제 LLM 스트리밍도 bot_logic이 담당
        for chunk in bot_logic.generate_answer_stream(prompt, docs, qa_chain):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # 마지막으로 봇의 답변을 세션에 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})
