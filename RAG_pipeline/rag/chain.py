from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(model_name: str = "gpt-5-mini"):
    """
    검색기(Retriever)가 가져온 문서(Context)와 사용자 질문(Question)을 입력받아
    최종 답변을 생성하는 체인입니다.
    """
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    # RAG 프롬프트: 철저하게 Context에 기반해서만 답변하도록 통제
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 항공 보안 및 각국 세관 통관 규정을 친절하게 안내하는 AI 도우미입니다.
        아래 제공된 [규정 문서]만을 바탕으로 사용자의 질문에 답변하세요.
        문서에 없는 내용은 절대 지어내지 말고, "규정에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
        
        답변 시, 사용자에게 신뢰감을 주기 위해 내용의 출처(한국 공공데이터, 미국 CBP 등)를 함께 언급해 주면 좋습니다.
        
        [규정 문서]
        {context}
        
        [사용자 질문]
        {question}
        
        친절하고 명확한 답변:
        """
    )
    
    # LCEL(LangChain Expression Language)를 활용한 파이프라인 구성
    # context 구성 로직은 app.py나 외부에서 묶어서 전달합니다.
    chain = prompt | llm | StrOutputParser()
    return chain
