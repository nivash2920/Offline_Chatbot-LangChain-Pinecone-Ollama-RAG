# ------------------------
# imports
# ------------------------
import os
from dotenv import load_dotenv
import ollama

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings  # base class for custom embeddings

# ------------------------
# load environment variables
# ------------------------
load_dotenv()

# ------------------------
# custom Ollama embeddings wrapper
# ------------------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            response = ollama.embed(model=self.model, input=text)
            vec = response.get("embeddings", response)
            # flatten nested embeddings if necessary
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

# ------------------------
# initialize Pinecone
# ------------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# ------------------------
# initialize embeddings and vector store
# ------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ------------------------
# retrieval
# ------------------------
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

query = "what do you mean by Back-test Your System?"
results = retriever.invoke(query)

# ------------------------
# display results
# ------------------------
print("RESULTS:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
