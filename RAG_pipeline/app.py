import streamlit as st
import time
from dotenv import load_dotenv

# 내부 모듈 임포트
from etl.loader import get_retriever
from rag.retriever import get_query_analyzer_chain, build_retriever_with_filters
from rag.chain import get_rag_chain

# 1. 환경 변수 (API KEY) 로드
load_dotenv()

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="한-미 통합 규정 AI 비서", page_icon="✈️", layout="centered")

st.title("✈️ 똑똑한 공항/세관 AI 비서")
st.markdown("""
이 챗봇은 **한국의 공공데이터(표 형식)**와 **미국 CBP/TSA 규정(Q&A 형식)**이  
하나의 통일된 스키마로 정규화(ETL)된 RAG 파이프라인 위에서 동작합니다.
""")

# --- 세션 초기화 (대화 기록, 엔진 등) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 캐싱을 이용해 RAG 관련 무거운 체인들을 딱 한 번만 불러옵니다.
@st.cache_resource
def init_rag_components():
    # 1. 데이터베이스(Chroma) 연결
    base_retriever = get_retriever()
    # 2. 질문 분석기
    query_analyzer = get_query_analyzer_chain()
    # 3. 답변 생성 체인
    qa_chain = get_rag_chain()
    
    return base_retriever, query_analyzer, qa_chain

try:
    base_retriever, query_analyzer, qa_chain = init_rag_components()
except Exception as e:
    st.error("RAG 파이프라인 초기화 중 에러가 발생했습니다. API 키나 DB 경로 상태를 확인해주세요.")
    st.stop()

# --- UI: 기존 채팅 기록 화면에 뿌리기 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 메인 채팅 로직 ---
# 사용자가 채팅창에 입력을 넣으면 작동합니다
if prompt := st.chat_input("예: 컵라면(육류스프 내장) 미국 갈 때 위탁으로 부쳐도 되나요?"):
    
    # 1. 사용자 질문을 세션 상태에 저장하고 화면에 즉시 띄움
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 봇 답변 생성 영역
    with st.chat_message("assistant"):
        # UI 응답성에 도움을 주는 "상태 창" 표시
        with st.status("🔍 사용자의 의도를 분석하고 규정을 검색 중입니다...", expanded=True) as status:
            
            # Step A: 의도 추출
            st.write("1️⃣ 질문 의도 파악 중...")
            intent = query_analyzer.invoke({"query": prompt})
            st.write(f"📝 *파악된 의도: 목적지='{intent.target_country}', 수하물형태='{intent.transport_method}', 핵심키워드='{intent.item_name}'*")
            
            # Step B: 필터 반영된 검색 수행
            st.write("2️⃣ 필터가 결합된 Vector DB 검색 중...")
            # 실제 DB에서는 필터를 결합한 검색기를 새로 만듦.
            vectorstore = base_retriever.vectorstore
            filtered_retriever = build_retriever_with_filters(vectorstore, intent)
            
            # 관련된 문서를 긁어옵니다
            docs = filtered_retriever.invoke(prompt)
            
            if not docs:
                status.update(label="⚠️ 관련된 규정을 찾지 못했습니다.", state="error")
                st.stop()
            
            # Context 문자열 조합
            context_text = "\n\n".join([f"[출처: {d.metadata.get('source', '알수없음')}]\n{d.page_content}" for d in docs])
            
            # UI 시각적 증명 (정규화된 메타데이터)
            st.write("3️⃣ 발견된 단일 스키마 규정(한/미 혼합):")
            for i, d in enumerate(docs, 1):
                st.caption(
                    f"💡 문서 {i}: {d.metadata.get('category')} | 제한: 기내({d.metadata.get('carry_on')}), "
                    f"위탁({d.metadata.get('checked_baggage')}), 미국입국({d.metadata.get('us_customs_admissibility')})"
                )
            
            status.update(label="✅ 검색 완료! 답변을 생성합니다.", state="complete", expanded=False)

        # Step C: LLM 최종 답변 생성(스트리밍 출력)
        message_placeholder = st.empty()
        full_response = ""
        
        # 스트리밍 방식(글자가 하나씩 쳐지는 효과)으로 답변 제공
        for chunk in qa_chain.stream({"context": context_text, "question": prompt}):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # 최종 완성된 답변 출력
        message_placeholder.markdown(full_response)
        
        # 마지막으로 봇의 답변을 세션에 저장 (다음 대화 흐름을 위해)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
