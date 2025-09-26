# -------------------------------
# Imports
# -------------------------------
import os
import time
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ollama
import ollama

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# -------------------------------
# Ollama embeddings wrapper for LangChain
# -------------------------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            response = ollama.embed(model=self.model, input=text)
            vec = response.get("embeddings", response)
            # Flatten nested embeddings if necessary
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

# -------------------------------
# Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
existing_indexes = [info["name"] for info in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,  # Ollama embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# -------------------------------
# Initialize embeddings and vector store
# -------------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# -------------------------------
# Load and split PDF documents
# -------------------------------
loader = PyPDFDirectoryLoader("documents/")  # folder containing PDFs
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.split_documents(raw_documents)

# -------------------------------
# Generate unique IDs
# -------------------------------
uuids = [f"id{i+1}" for i in range(len(documents))]

# -------------------------------
# Add documents to Pinecone
# -------------------------------
vector_store.add_documents(documents=documents, ids=uuids)

print(f"Successfully added {len(documents)} documents to Pinecone index '{PINECONE_INDEX_NAME}'")
