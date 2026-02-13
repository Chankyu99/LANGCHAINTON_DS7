# 🚀 LANGCHAINTON_DS7 협업 가이드

우리 팀의 원활한 협업을 위해 아래 규칙을 꼭 숙지하고 따라주세요!

## 📌 브랜치 목록 (본인 역할 확인)
- `feature/setup`: 찬규, 선우 (협업 환경, 로그 설계, 파이프라인 기초)
- `feature/rag`: 찬규, 선우 (RAG 파이프라인 개발)
- `feature/data`: 승희, 은영 (데이터 수집 및 전처리)
- `feature/prompt`: 승희, 은영 (프롬프트 테스트 및 최적화)
- `feature/ui`: 병곤 (카카오톡 연동, UI/UX 관련 코드)
- `docs`: 선우 (발표 자료, README 등 문서 작업)

---

## 🛠 1. 맨 처음 할 일 (최초 1회만)

자신의 담당 브랜치를 만들고 이동합니다.

```bash
git checkout -b 본인담당브랜치명
git push origin 본인담당브랜치명
```

## 💻 2. 매일 작업 시작 전

메인에 올라온 최신 코드를 내 컴퓨터로 가져온 뒤, 내 브랜치로 이동합니다.


```Bash
git checkout main
git pull origin main
git checkout 본인담당브랜치명
```

## 📤 3. 작업이 끝나고 저장(Push)할 때

```Bash
git add .
git commit -m "작업 내용 요약 (예: 데이터 수집 완료)"
git push origin 본인담당브랜치명
```

🤝 4. 합치기 (Pull Request)
깃허브 사이트의 Pull requests 탭에서 [New pull request] 클릭!

Reviewers: 함께 작업하는 파트너를 지정해주세요.

동료가 확인하고 승인하면 main에 합쳐집니다.

⚠️ 꼭 지켜주세요!

1. main 브랜치 직접 Push 절대 금지! (반드시 본인 브랜치 사용)

2. 커밋 메시지는 정성껏: "수정"보다는 "RAG 파이프라인 로직 개선"이 좋아요.

3. 작업 단위는 작게: 한꺼번에 올리지 말고 조금씩 자주 커밋하세요.

4. API 키 주의: .env 파일에 보관하고 절대 깃허브에 올리지 마세요.