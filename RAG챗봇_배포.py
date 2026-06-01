"""
=====================================================================
공정선거보도 안내서 RAG 챗봇 — Streamlit + Hugging Face Spaces 버전
=====================================================================

[이 파일이 하는 일]
이미 구축된 Chroma 벡터 DB(`chroma_election_db/`)를 로드해
브라우저에서 대화할 수 있는 RAG 챗봇입니다.

사용자가 질문을 입력하면:
  1) 질문을 Gemini Embedding으로 벡터화
  2) Chroma DB에서 가장 관련 높은 문서 조각(청크) 검색
  3) 검색된 내용 + 질문을 Gemini 2.5 Flash-Lite에 전달
  4) 모델이 출처(페이지)를 포함한 답변 생성
  5) Streamlit 화면에 멀티턴 대화로 표시

[전제 조건]
- `chroma_election_db/` 폴더가 이미 빌드되어 있어야 합니다.
- DB 빌드(문서 청킹+임베딩)는 별도 스크립트에서 수행했다고 가정합니다.

[기술 스택]
- 프론트엔드  : Streamlit
- 임베딩 모델 : Gemini Embedding 001 (768차원)
- 벡터 DB     : Chroma (사전 빌드된 영구 저장소)
- 생성 모델   : Gemini 2.5 Flash-Lite
- 배포 플랫폼 : Hugging Face Spaces

[설치 방법]
처음 한 번만 아래 명령으로 필요한 패키지를 설치합니다.
(가상환경(venv) 안에서 실행하는 것을 권장합니다.)

    pip install streamlit google-generativeai chromadb python-dotenv

또는 `requirements.txt` 파일에 다음을 적어두고 한 번에 설치할 수도 있습니다.

    streamlit
    google-generativeai
    chromadb
    python-dotenv

    → pip install -r requirements.txt

[실행 방법]
1) 같은 폴더에 `.env` 파일을 만들고 한 줄을 추가합니다.
       GEMINI_API_KEY=본인의_Gemini_API_키
2) 사전 빌드된 `chroma_election_db/` 폴더가 같은 경로에 있는지 확인합니다.
3) 아래 명령으로 앱을 띄웁니다(파이썬 직접 실행이 아니라 `streamlit run` 사용!).

       streamlit run RAG챗봇_배포.py

   브라우저가 자동으로 열리지 않으면 터미널에 표시된 http://localhost:8501 로 접속하세요.
=====================================================================
"""

# ─────────────────────────────────────────────
# 1. 라이브러리 가져오기
# ─────────────────────────────────────────────
import os
from pathlib import Path

import streamlit as st
import google.generativeai as genai
import chromadb

# dotenv는 로컬 개발 시 .env 파일에서 API 키를 불러올 때만 사용
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────
# 2. 전역 설정값
# ─────────────────────────────────────────────
DB_PATH = "./chroma_election_db"
# ↑ 이미 빌드된 Chroma DB 폴더 (이 폴더가 없으면 앱이 중단됩니다).

COLLECTION_NAME = "election_guide"
# ↑ DB 빌드 시 사용한 컬렉션 이름과 동일해야 합니다.

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash-lite"
EMBED_DIM = 768
TOP_K = 5


# ─────────────────────────────────────────────
# 3. Gemini API 초기화
# ─────────────────────────────────────────────
def init_gemini() -> bool:
    """
    Gemini API 키를 환경변수에서 읽어 설정합니다.

    키 출처:
      - 로컬 개발: 같은 폴더의 `.env` 파일이 `load_dotenv()`로 로드되어
        `os.environ`에 주입됩니다.
      - Hugging Face Spaces 배포: Space의 Settings → Secrets에 등록한 값이
        컨테이너 환경변수로 자동 주입됩니다.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.error(
            "❌ **GEMINI_API_KEY를 찾을 수 없습니다.**\n\n"
            "**로컬 개발 시:** 같은 폴더의 `.env` 파일에 "
            "`GEMINI_API_KEY=본인키` 한 줄을 추가하세요.\n\n"
            "**Hugging Face Spaces 배포 시:** Space의 Settings → Secrets에 "
            "`GEMINI_API_KEY` 항목을 추가하세요."
        )
        return False

    genai.configure(api_key=api_key)
    return True


# ─────────────────────────────────────────────
# 4. 질문 임베딩
# ─────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    """
    사용자 질문을 768차원 벡터로 변환합니다.

    DB 빌드 시 문서는 task_type="retrieval_document"로 임베딩되었으므로,
    질문은 짝이 되는 task_type="retrieval_query"를 사용합니다.
    이 비대칭 임베딩이 검색 정확도를 높여 줍니다.
    """
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
        output_dimensionality=EMBED_DIM
    )
    return result["embedding"]


# ─────────────────────────────────────────────
# 5. 기존 Chroma DB 로드
# ─────────────────────────────────────────────
# @st.cache_resource:
#   함수의 반환값(여기서는 Chroma 컬렉션 객체)을 메모리에 1회만 만들어 두고,
#   이후 호출에서는 다시 만들지 않고 같은 객체를 재사용하는 Streamlit 캐시.
#   사용자가 질문할 때마다 화면이 새로 그려지는데, DB 연결을 매번 다시 열면
#   느려지므로 이렇게 한 번만 로드합니다.
#   show_spinner=False → 캐시가 동작할 때 "Running..." 스피너를 띄우지 않음.
@st.cache_resource(show_spinner=False)
def get_collection():
    """
    사전 빌드된 Chroma 컬렉션을 로드합니다.

    이 함수는 DB를 새로 만들지 않습니다.
    `chroma_election_db/` 폴더와 컬렉션이 존재하지 않으면
    오류를 표시하고 앱을 중단합니다.
    """
    if not init_gemini():
        st.stop()

    if not Path(DB_PATH).exists():
        st.error(
            f"❌ Chroma DB 폴더를 찾을 수 없습니다: `{DB_PATH}`\n\n"
            "이 앱은 사전에 빌드된 벡터 DB가 있다고 가정합니다. "
            "DB 빌드 스크립트를 먼저 실행해 주세요."
        )
        st.stop()

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        st.error(
            f"❌ 컬렉션 `{COLLECTION_NAME}`을(를) 열 수 없습니다.\n\n"
            f"DB 경로(`{DB_PATH}`)와 컬렉션 이름을 확인해 주세요.\n\n"
            f"세부 오류: {e}"
        )
        st.stop()

    if col.count() == 0:
        st.error(
            f"❌ 컬렉션 `{COLLECTION_NAME}`이 비어 있습니다.\n\n"
            "DB 빌드 스크립트를 먼저 실행해 청크를 적재해 주세요."
        )
        st.stop()

    return col


# ─────────────────────────────────────────────
# 6. 관련 청크 검색
# ─────────────────────────────────────────────
def search(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    질문과 가장 유사한 청크 top_k개를 코사인 유사도로 검색합니다.

    Returns:
        [{"text": 청크내용, "page": 페이지번호, "score": 유사도}, ...]
    """
    q_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "text": doc,
            "page": meta.get("page", "?"),
            "score": round(1 - dist, 3)
        })
    return hits


# ─────────────────────────────────────────────
# 7. RAG 답변 생성
# ─────────────────────────────────────────────
def generate_answer(collection, query: str, chat_history: list[dict]) -> str:
    """
    검색된 문서 청크 + 대화 이력을 바탕으로 Gemini가 답변을 생성합니다.

    프롬프트 설계 원칙:
    1. "이 자료만 참고하라" → 환각(hallucination) 방지
    2. "[페이지 N] 출처 표기" → 신뢰성, 검증 가능성 확보
    3. "모르면 모른다고 하라" → 부정확한 정보 생성 억제
    4. temperature=0.2 → 창의적 답변보다 사실 기반 답변
    """
    hits = search(collection, query)

    context_parts = [
        f"[문서 {i}] (페이지 {h['page']}, 유사도={h['score']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    ]
    context = "\n\n".join(context_parts)

    system_prompt = """당신은 인터넷선거보도심의위원회의 「2026 공정선거보도 안내서」 전문 도우미입니다.
아래 [검색된 문서]만 참고하여 질문에 답하세요.

답변 규칙:
1. 각 주장 끝에 반드시 [페이지 N] 형식으로 출처를 표기하세요.
   예) 후보자 홍보물을 그대로 게재하면 공정성 위반에 해당합니다. [페이지 23]
2. 문서에 없는 내용은 추측하지 말고 "제공된 자료에서는 확인할 수 없습니다"라고 답하세요.
3. 이전 대화를 고려하여 자연스럽게 답하세요.
4. 답변은 한국어로 작성하세요.
5. 복잡한 내용은 번호 목록으로 정리하세요."""

    user_message = f"""{system_prompt}

[검색된 문서]
{context}

[질문]
{query}"""

    model = genai.GenerativeModel(
        GEN_MODEL,
        generation_config={"temperature": 0.2}
    )

    chat = model.start_chat(history=chat_history)
    response = chat.send_message(user_message)

    source_pages = sorted({h["page"] for h in hits})
    answer_text = response.text
    answer_text += f"\n\n---\n📚 **참고 페이지:** {source_pages}"

    return answer_text


# ─────────────────────────────────────────────
# 8. Streamlit UI 구성 (메인)
# ─────────────────────────────────────────────
def main():
    # ── 페이지 기본 설정 ──
    # set_page_config: 브라우저 탭에 보이는 제목/아이콘과 화면 너비를 정한다.
    #   layout="centered" → 본문이 가운데 좁게(채팅 UI에 적합). "wide"는 화면 가득.
    # ※ 반드시 스크립트 안에서 Streamlit 명령 중 가장 먼저 호출해야 한다.
    st.set_page_config(
        page_title="공정선거보도 RAG 챗봇",
        page_icon="🗳️",
        layout="centered"
    )

    st.title("🗳️ 공정선거보도 안내서 RAG 챗봇")
    st.caption(
        "인터넷선거보도심의위원회 「2026 공정선거보도 안내서」를 기반으로 답변합니다. "
        "답변에는 출처 페이지가 표시됩니다."
    )
    st.divider()

    # ── 대화 상태 초기화 ──
    # Streamlit은 사용자가 무언가를 누를 때마다 스크립트를 처음부터 다시 실행한다.
    # 그래서 일반 변수는 매번 사라지고, 대화 기록처럼 "유지되어야 하는 값"은
    # st.session_state(브라우저 세션 동안 살아 있는 딕셔너리)에 저장한다.
    # "messages"가 아직 없으면(=첫 실행) 빈 리스트로 만들어 둔다.
    if "messages" not in st.session_state:
        # 화면에 보여줄 대화 이력 (역할 + 텍스트)
        st.session_state.messages = []

    if "gemini_history" not in st.session_state:
        # Gemini API에 전달할 대화 이력 (Gemini 형식: role + parts)
        # 화면용과 API용 형식이 달라서 두 개를 따로 관리한다.
        st.session_state.gemini_history = []

    # ── 사이드바 영역 ──
    # `with st.sidebar:` 블록 안의 모든 위젯은 좌측 사이드바에 그려진다.
    # 블록을 벗어나면 다시 메인 영역으로 돌아간다(파이썬의 컨텍스트 매니저).
    with st.sidebar:
        st.header("📌 사용 안내")
        st.markdown("""
**이 챗봇은?**
- 105쪽 분량의 공정선거보도 안내서를 학습한 벡터 DB 사용
- 질문하면 관련 내용을 찾아 답변
- 답변마다 페이지 출처 표기

**예시 질문:**
        """)

        example_questions = [
            "인터넷선거보도심의위원회는 어떻게 구성되나요?",
            "후보자 홍보자료를 그대로 게재하면 어떻게 되나요?",
            "여론조사 결과 보도 시 주의사항은 무엇인가요?",
            "선거기간 중 특별제한 기간이 있나요?",
            "심의 신청은 어떻게 하나요?",
        ]

        # 예시 질문 버튼들.
        #   key=f"ex_{q}" → 같은 페이지에 여러 버튼이 있을 때 충돌하지 않도록
        #                   각 버튼에 고유 식별자를 부여한다(필수는 아니지만 권장).
        #   use_container_width=True → 버튼 폭을 사이드바 가로 폭에 맞춰 늘인다.
        # 사용자가 버튼을 누르면 즉시 다음 줄(채팅 입력창)을 건너뛰므로,
        # 클릭된 질문은 session_state에 "대기 중인 질문"으로 보관해 두었다가
        # 아래에서 꺼내 쓴다(버튼은 누른 그 순간의 재실행 중에만 True를 돌려준다).
        for q in example_questions:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.divider()

        # 대화 초기화 버튼: 두 이력 모두 비우고 화면을 즉시 다시 그린다.
        # st.rerun() → 지금 위치에서 스크립트를 처음부터 다시 실행시키는 명령.
        #              여기서는 빈 상태가 화면에 바로 반영되게 하기 위함.
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.gemini_history = []
            st.rerun()

    # ── 벡터 DB 로드 ──
    # `with st.spinner(...)`: 블록이 끝날 때까지 로딩 표시(스피너)를 보여준다.
    # 단, get_collection은 @st.cache_resource로 캐시되므로 두 번째부터는 거의 즉시 끝난다.
    with st.spinner("벡터 DB를 불러오는 중..."):
        collection = get_collection()

    # ── 이전 대화 다시 그리기 ──
    # 위에서 설명했듯 매 입력마다 스크립트가 처음부터 실행되므로,
    # 지금까지 쌓아 둔 messages를 순서대로 다시 화면에 출력해야 한다.
    # st.chat_message(role): "user"/"assistant" 말풍선 UI를 만들어 주는 컨테이너.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 이번 턴의 질문 결정 ──
    # pop("pending_question", None):
    #   사이드바 예시 버튼이 눌렸다면 거기에 담아 둔 질문을 꺼내면서 동시에 삭제.
    #   없으면 기본값 None을 돌려준다(다음 재실행 때 중복 실행 방지).
    # st.chat_input: 화면 하단에 고정된 채팅 입력창. 엔터를 치면 입력 문자열을, 비어 있으면 None을 반환.
    # `user_input or pending`:
    #   파이썬에서 `A or B`는 "A가 참 같은 값이면 A, 아니면 B"를 돌려준다.
    #   → 직접 타이핑이 우선, 없으면 예시 버튼 클릭값을 사용한다.
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("질문을 입력하세요... (예: 심의 기준이 무엇인가요?)")
    query = user_input or pending

    # 이번 재실행에서 질문이 있을 때만 답변 생성 절차로 들어간다.
    if query:
        # 사용자가 보낸 메시지를 즉시 화면에 그려 준다(반응성 ↑).
        with st.chat_message("user"):
            st.markdown(query)

        # 화면용 이력에 사용자 발화를 기록.
        st.session_state.messages.append({"role": "user", "content": query})

        # 어시스턴트 말풍선 영역을 미리 열고, 그 안에서 스피너를 띄우며 답변을 생성한다.
        # try/except로 감싸 API 오류가 나도 앱이 죽지 않고 안내 메시지로 대체되게 한다.
        with st.chat_message("assistant"):
            with st.spinner("🔍 관련 내용을 검색하고 답변을 생성하는 중..."):
                try:
                    answer = generate_answer(
                        collection,
                        query,
                        st.session_state.gemini_history
                    )
                except Exception as e:
                    answer = f"⚠️ 오류가 발생했습니다: {str(e)}\n\n잠시 후 다시 시도해 주세요."

            st.markdown(answer)

        # 두 종류의 이력 모두 업데이트:
        #   messages         → 다음 재실행 때 화면에 다시 그릴 용도
        #   gemini_history   → 다음 호출에서 Gemini가 맥락을 이어가도록 전달할 용도
        # Gemini 채팅 API의 형식은 {"role": "user"|"model", "parts": 텍스트}이다.
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.gemini_history.append({"role": "user", "parts": query})
        st.session_state.gemini_history.append({"role": "model", "parts": answer})


# ─────────────────────────────────────────────
# 9. 앱 실행 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
