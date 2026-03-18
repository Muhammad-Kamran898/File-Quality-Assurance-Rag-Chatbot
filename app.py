import streamlit as st
import os
import pandas as pd
import tempfile

from langchain_ollama import ChatOllama, OllamaEmbeddings
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

# Customizing the Initial App Landing Page
st.set_page_config(page_title="File QA ChatBot", page_icon="🤖")
st.title("File QA RAG Chatbot with Ollama 🤖")

# --- Hide Streamlit UI Styling ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache_resource(ttl="1h", show_spinner=False)
def configure_retriever(uploaded_files):
    embeddings_model = OllamaEmbeddings(
        model="nomic-embed-text", base_url="http://localhost:11434"
    )
    persist_dir = "./chroma_db_store"

    # 1. If the user uploaded new files, process and save them
    if uploaded_files:
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyMuPDFLoader(temp_filepath)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        doc_chunks = text_splitter.split_documents(docs)

        if not doc_chunks:
            st.error("No text could be extracted from the uploaded PDF.")
            st.stop()

        st.info("Building new database and saving to disk...")

        # Initialize empty Chroma DB pointing to our save folder
        vector_db = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings_model
        )

        # Progress bar logic for batch processing
        total_chunks = len(doc_chunks)
        batch_size = 100
        progress_text = f"Embedding {total_chunks} chunks. This might take a while..."
        my_bar = st.progress(0, text=progress_text)

        for i in range(0, total_chunks, batch_size):
            batch = doc_chunks[i : i + batch_size]
            vector_db.add_documents(batch)
            current_progress = min((i + batch_size) / total_chunks, 1.0)
            chunks_processed = min(i + batch_size, total_chunks)
            my_bar.progress(
                current_progress,
                text=f"Processed {chunks_processed} out of {total_chunks} chunks...",
            )

        my_bar.empty()
        return vector_db.as_retriever()

    # 2. If NO files were uploaded, try to load the existing database
    else:
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            st.success("Loaded existing database from disk!")
            vector_db = Chroma(
                persist_directory=persist_dir, embedding_function=embeddings_model
            )
            return vector_db.as_retriever()
        else:
            st.info("No existing database found. Please upload a PDF in the sidebar.")
            st.stop()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        # Capture the context from the main thread when initialized
        self.ctx = get_script_run_ctx()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Attach the context to the background thread before writing to the UI
        add_script_run_ctx(ctx=self.ctx)
        self.text += token
        self.container.markdown(self.text)


# Callback handler for processing and displaying top 3 document sources
class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, container):
        BaseCallbackHandler.__init__(self)
        self.container = container
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", "Unknown")

            metadata = {
                "source": source,
                "page": page,
                "content": d.page_content[:200] + "...",
            }
            idx = (source, page)
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            with self.container:
                st.markdown("**Sources:**")
                st.dataframe(
                    data=pd.DataFrame(self.sources[:3]), use_container_width=True
                )


# Creates UI element to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF Files", type=["pdf"], accept_multiple_files=True
)

# Creates a retriever object based on uploaded files
retriever = configure_retriever(uploaded_files)

# Creating a connection to Ollama LLM
llm = ChatOllama(model="llama3.2", temperature=0.1, base_url="http://localhost:11434")

# 1. Prompt to contextualize the user's question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# This creates a retriever that knows about the chat history!
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Prompt for the actual Question & Answering
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer as concise as possible."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 3. Combine them into the final RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
qa_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Store conversation history in Streamlit session state
streamlit_message_history = StreamlitChatMessageHistory(key="langchain_messages")

if len(streamlit_message_history.messages) == 0:
    streamlit_message_history.add_ai_message(
        "How can I help you with your documents today?"
    )

# Rendering current messages from Streamlit chat message history
for msg in streamlit_message_history.messages:
    st.chat_message(msg.type).write(msg.content)


# Handle new user input
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):
        # 1. Create the empty placeholder for the streaming text
        response_placeholder = st.empty()

        # 2. Create the container for the sources (so they print at the bottom)
        sources_container = st.container()

        # 3. Initialize BOTH of your callbacks
        stream_handler = StreamHandler(response_placeholder)
        pm_handler = PostMessageHandler(sources_container)

        # 4. Pass both callbacks into the config
        config = {"callbacks": [pm_handler, stream_handler]}

        # 5. Invoke the chain! The stream_handler will update the UI token by token.
        response = qa_rag_chain.invoke(
            {"input": user_prompt, "chat_history": streamlit_message_history.messages},
            config,
        )
