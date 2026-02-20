from dotenv import load_dotenv

from etl.extractor import extract_raw_data
from etl.transformer import build_transformer_chain, process_chunk_to_document
from etl.loader import build_vector_db

def run_etl_pipeline():
    # 1. 환경 변수(API 키 등) 로드
    load_dotenv()
    
    print("============================================================")
    print("  🚀 1단계: Extractor (데이터 추출)")
    print("============================================================")
    # 현재는 하드코딩된 한국어/영어 스니펫을 가져옵니다.
    raw_chunks = extract_raw_data()
    for i, chunk in enumerate(raw_chunks, 1):
        print(f"📄 Raw Chunk {i}: {chunk[:50]}...")
        
    print("\n============================================================")
    print("  🤖 2단계: Transformer (gpt-5-mini 기반 규정 추출 & 번역)")
    print("============================================================")
    chain = build_transformer_chain()
    
    transformed_docs = []
    for i, chunk in enumerate(raw_chunks, 1):
        print(f"🔄 Processing Chunk {i}...")
        doc = process_chunk_to_document(chunk, chain)
        transformed_docs.append(doc)
        print(f"   [결과] 물품명: {doc.metadata['item_name']} | 분류: {doc.metadata['category']} | 검색용 본문 길이: {len(doc.page_content)}")
        print(f"   [상세 조건(번역됨)]: {doc.page_content.split('규정 상세: ')[-1][:50]}...")
        
    print("\n============================================================")
    print("  🗄️ 3단계: Loader (Chroma DB 적재 및 다국어 임베딩)")
    print("============================================================")
    # Chroma DB에 document들을 적재합니다.
    vector_db = build_vector_db(transformed_docs)
    
    print("\n🎉 ETL 파이프라인 테스트 구축 및 적재가 완료되었습니다!")
    
    # [Self-Querying 시뮬레이션]
    # 실제 환경에서는 Self-Query Retriever가 이 필터를 LLM을 통해 자동 생성합니다.
    print("\n🔎 (테스트) 필터가 적용된 검색 시뮬레이션: 위탁 수하물이 금지된 전자기기 검색")
    test_retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 1,
            "filter": {
                "$and": [
                    {"category": "전자기기"},
                    {"checked_baggage": "금지"}
                ]
            }
        }
    )
    results = test_retriever.invoke("기내에 가져가도 되는 보조 배터리")
    if results:
        print(f"   -> 검색된 문서: {results[0].metadata['item_name']} (출처: {results[0].metadata['source']})")
    else:
        print("   -> 조건에 맞는 문서가 없습니다.")

if __name__ == "__main__":
    run_etl_pipeline()
