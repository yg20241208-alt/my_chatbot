# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st

# -------------------------------------------------------------------
# ✅ sqlite3 호환 (Streamlit Cloud 등 일부 환경에서 Chroma가 sqlite3 빌드 이슈를 일으킬 때 대응)
#    - 반드시 Chroma/ChromaDB import "이전"에 실행되어야 합니다.
# -------------------------------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # pysqlite3가 없거나 교체가 불필요한 환경이면 그대로 진행
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma

# -------------------------------------------------------------------
# ✅ API Key (Str돌.pdf"

uploaded = st.file_uploader("PDF를 업로드하거나, 기본 PDF로 실행하세요.", type=["pdf"])
pdf_path = None

if uploaded is not None:
    # 업로드 파일은 임시로 저장 후 사용
    tmp_dir = Path(".streamlit_tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = str(tmp_dir / uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())
else:
    # 기본 파일이 레포에 포함돼 있다면 상대경로로 접근
    if os.path.exists(DEFAULT_PDF):
        pdf_path = DEFAULT_PDF

if not pdf_path:
    st.info("먼저 PDF를 업로드하시거나, 레포에 기본 PDF 파일을 추가해주세요.")
    st.stop()

rag_chain = initialize_chain(option, pdf_path)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 기존 대화 렌더링
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 입력
if prompt_message := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response.get("answer", "")
            st.write(answer)

            with st.expander("참고 문서 확인"):
                for doc in response.get("context", []):
                    src = doc.metadata.get("source", "source")
                    st.markdown(src, help=doc.page_content)



