📄 RAG Document QA Chatbot (Streamlit + LangChain + Ollama)
An interactive, locally-hosted Retrieval-Augmented Generation (RAG) chatbot application. This tool allows users to upload documents (PDFs) and seamlessly chat with their data using local Large Language Models (LLMs) via Ollama. Built with a modern Python stack featuring Streamlit for the UI and LangChain for the AI orchestration.

✨ Features
Conversational AI: Chat naturally with your documents using context-aware responses.

Privacy-First (Local LLMs): Powered by Ollama, meaning your sensitive documents are never sent to external cloud APIs like OpenAI (unless configured to do so).

Real-time Streaming: Token-by-token response streaming for a fluid, ChatGPT-like user experience.

Chat History Memory: The model remembers the context of previous questions within the session.

Robust Document Processing: Utilizes PyMuPDF for accurate text extraction and ChromaDB for highly efficient vector storage.

🛠️ Tech Stack
Frontend: Streamlit

AI Orchestration: LangChain (Core, Community, and Classic components)

Local Models: Ollama

Vector Database: Chroma

Document Parsing: PyMuPDF

🧠 How It Works (The RAG Pipeline)
Document Loading: The user uploads a PDF file through the Streamlit interface.

Splitting & Chunking: The application breaks the document down into smaller, semantic chunks using RecursiveCharacterTextSplitter so the AI can digest it easily.

Embedding: The text chunks are converted into numerical vectors using OllamaEmbeddings.

Vector Storage: These embeddings are stored locally in a Chroma database for lightning-fast semantic search.

Retrieval & Generation: When a user asks a question, the app searches the Chroma DB for the most relevant document chunks, packages them with the user's prompt and chat history, and sends them to the local LLM to generate a precise answer.

🚀 Getting Started
Prerequisites
Python 3.9+ installed on your machine.

Ollama installed and running locally. (Download Ollama here).

Pull your preferred local model via your terminal (e.g., Llama 3):

Bash
ollama run llama3
Installation
Clone the repository:

Bash
git clone https://github.com/YourUsername/Your-Repo-Name.git
cd Your-Repo-Name
Create a virtual environment (Recommended):

Bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
Install the required dependencies:

Bash
pip install -r requirements.txt
Note: Ensure your environment supports modern LangChain components (langchain-core, langchain-ollama, langchain-chroma, and langchain-classic for legacy LCEL chains).

Running the App
Launch the Streamlit interface by running the following command in your terminal:

Bash
streamlit run app.py
This will open a new tab in your default web browser (usually at http://localhost:8501).

⚠️ Troubleshooting
Windows Path Length Errors: If you encounter a [Errno 2] No such file or directory error during installation on Windows, your folder path name is likely exceeding Windows' 260-character limit. Move the repository closer to your root drive (e.g., C:\Projects\) or enable long paths in the Windows Registry.

Missing ScriptRunContext: If the UI crashes when the LLM attempts to stream a response, ensure that get_script_run_ctx() is properly attaching Streamlit's session state to the background LangChain callback thread.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute."# File-Quality-Assurance-Rag-Chatbot" 
