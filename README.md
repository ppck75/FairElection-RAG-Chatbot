---
title: 공정선거보도 RAG 챗봇
emoji: 🗳️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: RAG챗봇_배포.py
pinned: false
license: mit
---

# 🗳️ 공정선거보도 안내서 RAG 챗봇

인터넷선거보도심의위원회의 **「2026 공정선거보도 안내서」**(약 105쪽)를 학습한
검색-증강 생성(RAG) 챗봇입니다. 질문에 대해 안내서 본문에서 관련 내용을 찾아
**출처 페이지 번호와 함께** 답변합니다.

## 사용 방법

1. 사이드바의 예시 질문 버튼을 누르거나, 하단 입력창에 직접 질문 입력
2. 답변과 함께 표시되는 `[페이지 N]` 출처를 원문에서 확인
3. 이어지는 질문은 이전 맥락을 고려해 자연스럽게 답변 (멀티턴 대화)
4. 대화를 처음부터 시작하려면 사이드바의 **🗑️ 대화 초기화** 클릭

## 기술 스택

| 구성 요소 | 사용 기술 |
|---|---|
| 프론트엔드 | Streamlit |
| 임베딩 모델 | Gemini Embedding 001 (768차원) |
| 벡터 DB | Chroma (사전 빌드된 영구 저장소) |
| 생성 모델 | Gemini 2.5 Flash-Lite |
| 배포 | Hugging Face Spaces |

## 동작 원리

```
질문 입력
   ↓
Gemini Embedding으로 벡터화 (task_type=retrieval_query)
   ↓
Chroma DB에서 코사인 유사도 Top-5 청크 검색
   ↓
검색 결과 + 시스템 프롬프트 + 대화 이력 → Gemini 2.5 Flash-Lite
   ↓
출처 페이지가 표시된 한국어 답변 생성
```

## 환경 변수

Space의 **Settings → Variables and secrets**에 아래 항목 등록 필요:

- `GEMINI_API_KEY` — Google AI Studio에서 발급 (https://aistudio.google.com/apikey)

## 로컬에서 실행하기

1. 의존 패키지 설치 (가상환경 권장)

   ```bash
   pip install -r requirements.txt
   ```

2. 같은 폴더에 `.env` 파일을 만들고 한 줄 추가

   ```
   GEMINI_API_KEY=본인의_Gemini_API_키
   ```

3. 사전 빌드된 `chroma_election_db/` 폴더가 같은 경로에 있는지 확인

4. Streamlit으로 앱 실행 (파이썬 직접 실행이 아님에 유의)

   ```bash
   streamlit run RAG챗봇_배포.py
   ```

   브라우저가 자동으로 열리지 않으면 http://localhost:8501 로 접속하세요.

## 파일 구조

```
.
├── RAG챗봇_배포.py        # 메인 앱 (Streamlit 진입점)
├── requirements.txt       # 의존 패키지
├── README.md              # 이 파일
└── chroma_election_db/    # 사전 빌드된 벡터 DB
    ├── chroma.sqlite3
    └── <UUID 폴더>/
```

벡터 DB는 별도 빌드 스크립트로 미리 생성되어 있어야 하며,
앱 실행 시에는 청킹·임베딩 작업을 수행하지 않습니다.

## 주의 사항

- 본 챗봇의 답변은 안내서 본문에 근거하지만, 최종 판단과 인용은 **원문 확인** 후 사용해 주세요.
- 무료 티어 환경에서는 일정 시간 미사용 시 Space가 슬립 상태가 되어 첫 요청에 30초가량 걸릴 수 있습니다.
- Gemini API 호출에는 별도의 사용량/요금 정책이 적용됩니다.
