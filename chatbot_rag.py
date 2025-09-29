# ------------------- IMPORTS -------------------
import os
from dotenv import load_dotenv
import streamlit as st

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import ollama
from langchain.embeddings.base import Embeddings

# ------------------- LOAD ENV -------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# ------------------- STREAMLIT TITLE -------------------
st.title("Nivash_Chatbot")

# ------------------- PINECONE SETUP -------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ------------------- CUSTOM OLLAMA EMBEDDINGS -------------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            response = ollama.embed(model=self.model, input=text)
            vec = response.get("embeddings", response)
            if isinstance(vec[0], list):
                vec = [float(v) for sublist in vec for v in sublist]
            else:
                vec = [float(v) for v in vec]
            all_embeddings.append(vec)
        return all_embeddings

    def embed_query(self, text):
        response = ollama.embed(model=self.model, input=text)
        vec = response.get("embeddings", response)
        if isinstance(vec[0], list):
            vec = [float(v) for sublist in vec for v in sublist]
        else:
            vec = [float(v) for v in vec]
        return vec

# ------------------- INITIALIZE VECTOR STORE -------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ------------------- INITIALIZE CHAT HISTORY -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are a personal assistant who has all my professional details.."))

# ------------------- DISPLAY CHAT HISTORY -------------------
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ------------------- CHAT INPUT -------------------
prompt = st.chat_input("Welcome! How may I help you?")

if prompt:
    # Add user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Initialize retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    # Retrieve relevant documents
    docs = retriever.invoke(prompt)
    docs_text = "\n\n".join(d.page_content for d in docs[:5])  # limit to 5 chunks

    # System prompt with context
    system_prompt = f"""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {docs_text}"""

    # Initialize Ollama chat model
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=1)

    # Prepare messages and invoke LLM
    messages_to_llm = [SystemMessage(system_prompt), HumanMessage(prompt)]
    result = llm.invoke(messages_to_llm).content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))
