# AI Document Q&A System

🔗 **Demo:** [https://ai-document-q-a-system.streamlit.app/](https://ai-document-q-a-system.streamlit.app/)

A production-ready RAG (Retrieval-Augmented Generation) application that enables intelligent question-answering over PDF documents. Users can upload any PDF document and engage in natural language conversations to extract insights, ask follow-up questions, and receive contextually accurate answers with source citations.

## Overview

This application implements a dynamic RAG system that processes PDF documents, creates semantic embeddings, and uses advanced language models to answer questions based on document content. The system understands natural language, handles follow-up questions intelligently, and provides detailed source citations for transparency.

## Features

- **Dynamic PDF Upload**: Upload any PDF document through an intuitive web interface
- **Intelligent Document Processing**: Automatic text extraction, chunking, and embedding generation
- **Semantic Search**: Vector-based retrieval using FAISS for finding relevant document sections
- **Natural Language Understanding**: Handles follow-up questions and conversational context
- **Adaptive Responses**: Provides concise answers for simple questions and detailed explanations for complex queries
- **Conversation Memory**: Maintains context across multiple questions in a session
- **Modern UI**: Clean, user-friendly Streamlit interface with real-time processing feedback

## Architecture

The system follows a RAG (Retrieval-Augmented Generation) architecture with the following components:

### 1. Document Processing Pipeline
- **PDF Loading**: Extracts text from uploaded PDF documents using PyPDFLoader
- **Text Chunking**: Splits documents into semantically meaningful chunks (1000 characters with 200 character overlap) using RecursiveCharacterTextSplitter
- **Embedding Generation**: Converts text chunks into high-dimensional vector embeddings using HuggingFace sentence transformers

### 2. Vector Store
- **FAISS Index**: Stores document embeddings in a FAISS vector database for efficient similarity search
- **Persistent Storage**: Vector store is saved locally to avoid re-processing documents
- **Semantic Retrieval**: Retrieves top-k most relevant document chunks based on query similarity

### 3. Retrieval Chain
- **Query Embedding**: Converts user questions into embeddings using the same model
- **Similarity Search**: Finds the 5 most relevant document chunks using cosine similarity
- **Follow-up Detection**: Intelligently detects follow-up questions and uses previous conversation context for better retrieval

### 4. Generation Chain
- **Prompt Engineering**: Uses context-aware prompts that adapt based on question type (simple vs. complex)
- **LLM Integration**: Leverages Groq's Llama 3.1 8B Instant model for fast, accurate responses
- **Context Injection**: Passes retrieved document chunks as context to the LLM
- **Answer Synthesis**: Generates answers strictly based on provided context, with fallback messages when information is unavailable

### 5. Conversation Management
- **Session State**: Maintains conversation history using Streamlit's session state
- **Follow-up Handling**: Detects follow-up questions using natural language patterns and conversation history
- **Context Preservation**: Passes previous questions and answers to maintain conversational flow

## Technology Stack

### Frontend & Framework
- **Streamlit**: Web application framework for building the interactive UI
- **Python 3.9+**: Core programming language

### RAG & LLM Framework
- **LangChain**: Orchestration framework for RAG pipeline
- **LangChain Community**: Community integrations for document loaders and vector stores
- **Groq**: High-performance LLM inference using Llama 3.1 8B Instant model

### Embeddings & Vector Search
- **HuggingFace**: 
  - `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings
  - HuggingFace Inference API for cloud-based embeddings (with local fallback)
- **FAISS**: Facebook AI Similarity Search for efficient vector similarity operations

### Document Processing
- **PyPDF**: PDF text extraction and processing
- **RecursiveCharacterTextSplitter**: Intelligent text chunking with overlap

### Utilities
- **python-dotenv**: Environment variable management
- **Logging**: Comprehensive logging for debugging and monitoring

## How It Works

1. **Document Upload**: User uploads one or more PDF files through the web interface
2. **Processing**: System extracts text, splits into chunks, and generates embeddings
3. **Indexing**: Embeddings are stored in a FAISS vector database
4. **Query**: User asks a question in natural language
5. **Retrieval**: System finds the most relevant document chunks using semantic search
6. **Generation**: LLM synthesizes an answer from retrieved context
7. **Response**: Answer is displayed with source citations and metadata
8. **Follow-up**: System maintains conversation context for natural follow-up questions

## Key Design Decisions

- **Adaptive Answer Depth**: Prompt templates adjust response style based on question complexity
- **Robust Error Handling**: Graceful fallbacks for API failures and missing information
- **Efficient Caching**: LLM and retriever objects are cached to improve performance
- **User Experience**: Real-time feedback during processing, clear error messages, and intuitive UI
- **Privacy-First**: Documents are processed in-memory with optional local persistence


