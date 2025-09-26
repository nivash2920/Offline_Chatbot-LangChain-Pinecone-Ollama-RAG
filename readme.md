<h1>Retrieval Augmented Generation (RAG) with Streamlit, LangChain and Pinecone</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<ol>
  <li>
    Clone the repository:
    <pre>
git clone https://github.com/nivash2920/Offline_Chatbot-LangChain-Pinecone-Ollama-RAG.git
cd Offline_Chatbot-LangChain-Pinecone-Ollama-RAG
    </pre>
  </li>
  <li>
    Create a virtual environment:
    <pre>python -m venv venv</pre>
  </li>
  <li>
    Activate the virtual environment:
    <pre>
venv\Scripts\Activate   (on Windows)
source venv/bin/activate  (on Mac/Linux)
    </pre>
  </li>
  <li>
    Install libraries:
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li>
    Create accounts:
    <ul>
      <li>Pinecone: <a href="https://www.pinecone.io/">https://www.pinecone.io/</a></li>
      <li>OpenAI API key: <a href="https://platform.openai.com/api-keys">https://platform.openai.com/api-keys</a></li>
    </ul>
  </li>
  <li>
    Add API keys to .env file:
    <ul>
      <li>Rename <code>.env.example</code> to <code>.env</code></li>
      <li>Add your Pinecone and OpenAI API keys to the <code>.env</code> file</li>
    </ul>
  </li>
</ol>

<h2>Executing the scripts</h2>
<ol>
  <li>Open a terminal in VS Code</li>
  <li>Run the scripts:
    <pre>
python sample_ingestion.py
python sample_retrieval.py
python ingestion.py
python retrieval.py
streamlit run chatbot_rag.py
    </pre>
  </li>
</ol>
