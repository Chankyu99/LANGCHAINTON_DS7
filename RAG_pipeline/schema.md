# 🚀 '기내뭐돼' 최종 RAG 파이프라인 계획안

## 1단계: 오프라인 데이터 적재 (Data Ingestion & Indexing)

- 데이터 소스: 제공받은 index_docstore_export.jsonl 파일을 그대로 활용.

- 임베딩: OpenAI의 text-embedding-3-small 또는 유사한 성능의 임베딩 모델을 사용하여 page_content 텍스트를 벡터로 변환.
+2

- 메타데이터 분리 저장: recommended_metadata 안의 값들을 추출하여 벡터 DB(예: Chroma, Pinecone)의 메타데이터 필드에 정확히 매핑. 한국은 cabin_decision과 checked_decision으로 , 미국은 decision과 domain으로 구조화되어 있어 필터링에 아주 유리함

## 2단계: 대화 상태 관리 및 의도 파악 (Router & Slot Filling)

- 필수 슬롯 추출: 사용자의 질문에서 출발지, 도착지, 물품명을 추출.

- 누락 정보 대응: 물품명이나 노선이 빠졌다면 즉시 되묻기 로직 실행.

- 상세 속성 패스: 용량(Wh, ml) 등 상세 속성이 없어도 핑퐁 대화를 줄이기 위해 추가 질문 없이 즉시 다음 단계로 통과.

- 예외 처리: 시스템이 지원하지 않는 노선이 들어오면 에러 메시지 출력 후 대기.

## 3단계: 쿼리 재작성 및 하이브리드 검색 (Rewriter & Retriever)

- 용어 정규화: 사용자의 구어체나 은어(예: 빠떼리)를 공식 데이터셋의 전문 용어(item)로 변환.

- 교차 메타데이터 검색 (Self-Querying): 확정된 노선을 바탕으로 필터 조건을 자동 생성하여 Vector DB 검색. 한국 출발, 미국 도착이라면 jurisdiction: KR 데이터와 jurisdiction: US  데이터를 동시에 정확하게 끄집어냄.


## 4단계: 최종 판정 및 답변 생성 (Judge & Generator)

- 엄격한 판정 룰: 검색된 한국/미국 규정 중 단 하나라도 prohibited(금지) 상태라면 최종 답변은 무조건 [반입 불가]로 판정.


- 조건부 범위 안내: 메타데이터가 conditional 또는 declaration_required 인 경우, 무작정 안 된다고 하지 않고 page_content의 상세내용을 참조하여 허용되는 범위와 신고 절차를 시니어 층도 이해하기 쉬운 친절한 대화체로 풀어서 안내.

- 환각 방지 (우회 로직): 검색된 데이터에 없는 물품은 지어내지 않고, 항공사에 직접 문의하도록 가이드 메시지 출력.