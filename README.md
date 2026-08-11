---
title: Fair Election Reporting RAG Chatbot
emoji: 🗳️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: RAG챗봇_배포.py
pinned: false
license: mit
---

# 🗳️ Fair Election Reporting Guide RAG Chatbot

This retrieval-augmented generation (RAG) chatbot uses the Internet Election News
Deliberation Commission's approximately 105-page **2026 Fair Election Reporting Guide**.
It finds relevant information in the guide and answers questions **with source page numbers**.

## How to Use

1. Select a sample question in the sidebar, or enter your own question in the input field at the bottom.
2. Verify the `[페이지 N]` citation shown with the answer against the original guide.
3. Ask follow-up questions; the chatbot uses the previous context to respond naturally (multi-turn conversation).
4. To start over, click **🗑️ 대화 초기화** in the sidebar.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embedding model | Gemini Embedding 001 (768 dimensions) |
| Vector DB | Chroma (prebuilt persistent store) |
| Generative model | Gemini 2.5 Flash-Lite |
| Deployment | Hugging Face Spaces |

## How It Works

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

## Environment Variables

Add the following entry under **Settings → Variables and secrets** in the Space:

- `GEMINI_API_KEY` — Obtain from Google AI Studio (https://aistudio.google.com/apikey)

## Run Locally

1. Install the dependencies (a virtual environment is recommended).

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the same directory and add the following line.

   ```
   GEMINI_API_KEY=본인의_Gemini_API_키
   ```

3. Make sure the prebuilt `chroma_election_db/` directory is in the same location.

4. Run the app with Streamlit rather than invoking Python directly.

   ```bash
   streamlit run RAG챗봇_배포.py
   ```

   If the browser does not open automatically, open http://localhost:8501 in your browser.

## File Structure

```
.
├── RAG챗봇_배포.py        # 메인 앱 (Streamlit 진입점)
├── requirements.txt       # 의존 패키지
├── README.md              # 이 파일
└── chroma_election_db/    # 사전 빌드된 벡터 DB
    ├── chroma.sqlite3
    └── <UUID 폴더>/
```

The vector DB must be generated in advance with a separate build script.
The app does not perform chunking or embedding at runtime.

## Notes

- The chatbot's answers are based on the guide, but verify the **original guide** before using them for final decisions or citations.
- On the free tier, the Space may go to sleep after a period of inactivity, so the first request may take approximately 30 seconds.
- Separate usage limits and pricing policies apply to Gemini API calls.
