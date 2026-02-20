import streamlit as st
from bot_logic import ask_to_ai

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="스마트 항공 보안/세관 챗봇", 
    page_icon="🧳", 
    layout="centered"
)

st.title("✈️스마트 수하물 및 세관 규정 안내")
st.markdown("""
출발지, 도착지, 그리고 궁금한 물품을 함께 말씀해 주세요.
* **예시 질문:** "한국에서 뉴욕으로 가는데 미숫가루랑 라면 챙겨도 돼?"
""")

# ==========================================
# 2. 채팅 세션(Session State) 초기화
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 어디에서 어디로 가시나요? 어떤 물건이 궁금하신지 물어보세요! 😊"}
    ]

# [추가] 여행 정보를 기억할 바구니 생성
if "travel_info" not in st.session_state:
    st.session_state.travel_info = {"departure": "", "destination": "", "item": ""}

# ==========================================
# 3. 이전 대화 내역 렌더링
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. 사용자 입력 및 AI 답변 처리
# ==========================================
if prompt := st.chat_input("여기에 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("규정을 검색하고 있습니다... 🔍"):
            
            # [수정] 세션에 저장된 travel_info를 함께 보냅니다.
            answer, retrieved_docs, updated_info = ask_to_ai(prompt, st.session_state.travel_info)
            
            # [수정] AI가 업데이트해준 정보를 다시 세션에 저장합니다.
            st.session_state.travel_info = updated_info
            
            st.markdown(answer)
            
            # 4-3. 참고 문서(출처) UI 구성 (문서가 있을 때만 표시)
            if retrieved_docs:
                with st.expander("📚 참고한 공식 규정 원문 및 출처 보기"):
                    for doc in retrieved_docs:
                        meta = doc.metadata
                        jurisdiction = meta.get('jurisdiction', 'Unknown')
                        item_name = meta.get('item', '해당 품목')
                        source_hint = meta.get('source_hint', '출처 정보 없음')
                        
                        # 메타데이터 키 차이(evidence_url vs evidence_url_primary) 방어 로직
                        evidence_url = meta.get('evidence_url', meta.get('evidence_url_primary', '#'))
                        
                        # 국가별 직관적인 아이콘 표시 (국가 추가될 경우 수정)
                        icon = "🇰🇷" if jurisdiction == "KR" else "🇺🇸" if jurisdiction == "US" else "🌐"
                        
                        st.markdown(f"**{icon} [{jurisdiction}] {source_hint}**")
                        st.markdown(f"- 🔗 [{item_name} 관련 공식 규정 링크]({evidence_url})")
                        st.divider() # 항목 간 구분선
                        
        # 4-4. AI 답변을 세션에 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})