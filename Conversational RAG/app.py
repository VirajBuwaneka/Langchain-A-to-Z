import streamlit as st
from dotenv import load_dotenv
import os
from uuid import uuid4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ✅ Chroma (for LangChain 1.0.x)
from langchain_community.vectorstores import Chroma

# ✅ Correct imports for LangChain 1.0.x
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ----------------------------------------------------------------------------
# ✅ ENV + API KEY
# ----------------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PERSIST_DIR = "vector_store"


@st.cache_resource
def load_vectordb():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectordb


@st.cache_resource
def build_rag_chain():

    vectordb = load_vectordb()
    retriever = vectordb.as_retriever()

    llm = ChatOpenAI(model="gpt-4.1")

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user question using previous chat history only if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    system_prompt = (
        "You are an intelligent chatbot. Use the context below to answer.\n"
        "If you don't know, just say you don't know. 😊\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain


# ----------------------------------------------------------------------------
# ✅ Session-based Memory Store
# ----------------------------------------------------------------------------

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    build_rag_chain(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# ----------------------------------------------------------------------------
# ✅ Streamlit UI
# ----------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔍 RAG Chatbot with Memory (History Aware)")


if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid4())
    st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    bot_reply = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
