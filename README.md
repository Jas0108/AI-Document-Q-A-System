# Document Q&A System

A production-ready RAG (Retrieval-Augmented Generation) application for question-answering over documents.

## Features

- 📄 PDF document processing
- 🔍 Semantic search using vector embeddings
- 💬 AI-powered Q&A using Groq LLM
- 💾 Persistent vector store
- 📊 Source citations with metadata
- 🎨 Modern, user-friendly interface

## Setup

### Prerequisites

- Python 3.9 or higher
- Groq API key ([Get one here](https://console.groq.com))
- **Optional**: Google Generative AI API key (only needed if using Google embeddings)

### Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd Gemma
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   On Windows:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Edit `.env` and add your API keys:
     ```
     GROQ_API_KEY=your_actual_groq_key_here
     EMBEDDING_PROVIDER=huggingface  # Use "huggingface" (free) or "google" (requires API key)
     ```
   - **Note**: The app uses **free HuggingFace embeddings by default** (no API key needed). 
     If you want to use Google embeddings, set `EMBEDDING_PROVIDER=google` and add `GOOGLE_API_KEY`.

6. **Add documents (if needed)**
   - Place PDF files in the `us_census/` directory
   - Or update `DOCUMENTS_PATH` in `app.py` to point to your documents

## Running the Application

1. **Make sure virtual environment is activated**
   
   On Windows:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

2. **Run Streamlit**
   ```bash
   streamlit run app.py
   ```

3. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't, manually navigate to that URL

## Usage

1. **Upload PDFs**: Use the file uploader to upload one or more PDF documents
2. **Process Documents**: Click "Process Documents" to create embeddings from your PDFs
3. **Ask Questions**: Type your question in the chat input at the bottom
4. **View Sources**: Expand "Source Documents" to see where the answer came from
5. **Follow-up Questions**: Ask natural follow-up questions - the system understands context

## Project Structure

```
Gemma/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example           # Example environment file
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── us_census/            # PDF documents directory
└── storage/              # Vector store persistence (auto-created)
    └── vector_store/      # Saved embeddings
```

## Technology Stack

- **Streamlit** - Web interface
- **LangChain** - RAG framework
- **Groq** - LLM inference (Llama 3.1 8B Instant)
- **Embeddings**:
  - **HuggingFace** (default) - Free embeddings using sentence-transformers or Inference API
  - **Google Generative AI** (optional) - Cloud-based embeddings (requires API key)
- **FAISS** - Vector database

## Troubleshooting

### Import Errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python --version` (should be 3.9+)

### API Key Errors
- Verify `.env` file exists and contains `GROQ_API_KEY`
- For Google embeddings: Check `GOOGLE_API_KEY` is set if using `EMBEDDING_PROVIDER=google`
- **Tip**: Use `EMBEDDING_PROVIDER=huggingface` for free embeddings (no Google API key needed)

### Quota/429 Errors (Google Embeddings)
- If you see "quota exceeded" errors, the app will automatically switch to free HuggingFace embeddings
- To use HuggingFace by default, set `EMBEDDING_PROVIDER=huggingface` in your `.env` file

### Document Loading Errors
- Ensure `us_census/` directory exists
- Verify PDF files are in the directory
- Check file permissions

### Vector Store Issues
- Delete `storage/` folder to force re-processing
- Check disk space availability

## Deployment

### Streamlit Cloud (Recommended)

Deploy your app to Streamlit Cloud for free! See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed step-by-step instructions.

**Quick Steps:**
1. Push your code to GitHub (make repository public)
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add API keys in Streamlit Cloud Secrets:
   - `GROQ_API_KEY`
   - `HUGGINGFACE_API_KEY` (optional)
5. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

For detailed instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## License

MIT
