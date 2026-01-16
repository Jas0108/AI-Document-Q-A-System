import os
# Fix OpenMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration constants
DOCUMENTS_PATH = "./us_census"
VECTOR_STORE_PATH = "./storage/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GROQ_MODEL = "llama-3.1-8b-instant"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NUM_RETRIEVED_DOCS = 5  # Increased for better context retrieval

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_doc_set" not in st.session_state:
    st.session_state.current_doc_set = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


class HuggingFaceAPIEmbeddings(Embeddings):
    """HuggingFace embeddings using the InferenceClient (new API)"""
    
    def __init__(self, api_key: str, model_name: str = HUGGINGFACE_MODEL):
        self.api_key = api_key.strip().strip('"').strip("'")
        self.model_name = model_name
        self.client = InferenceClient(token=self.api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            results = []
            for text in texts:
                embedding = self.client.feature_extraction(text, model=self.model_name)
                if isinstance(embedding, list):
                    results.append(embedding)
                else:
                    results.append(list(embedding) if hasattr(embedding, '__iter__') else [float(embedding)])
            return results
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise ValueError(f"API request failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


def get_embeddings():
    """Get embeddings using HuggingFace - tries API first, falls back to local if API unavailable"""
    if HUGGINGFACE_API_KEY:
        try:
            embeddings = HuggingFaceAPIEmbeddings(
                api_key=HUGGINGFACE_API_KEY,
                model_name=HUGGINGFACE_MODEL
            )
            test_result = embeddings.embed_query("test")
            if test_result and len(test_result) > 0:
                logger.info(f"✅ Using HuggingFace API! Embedding dimension: {len(test_result)}")
                return embeddings
        except Exception as e:
            logger.warning(f"HuggingFace API not available: {e}")
            logger.info("Using local embeddings instead (free, cached, reliable)")
    
    logger.info("Using local HuggingFace embeddings (model cached, no download needed after first use)")
    return HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def validate_api_keys() -> tuple[bool, Optional[str]]:
    """Validate that required API keys are present"""
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not groq_key:
        return False, "GROQ_API_KEY is missing. Please set it in your .env file."
    
    return True, None


def save_vector_store(vectors, path: str) -> None:
    """Save vector store to disk"""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vectors.save_local(path)
        logger.info(f"Vector store saved to {path}")
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
        raise


def load_vector_store(embeddings, path: str):
    """Load vector store from disk"""
    try:
        if os.path.exists(path):
            vectors = FAISS.load_local(
                path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {path}")
            return vectors
        return None
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    if not uploaded_files:
        raise ValueError("No files uploaded")
    
    temp_dir = None
    try:
        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        # Save uploaded files temporarily
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Load all PDFs
        all_docs = []
        for file_path in file_paths:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise ValueError(f"Failed to load {os.path.basename(file_path)}: {str(e)}")
        
        if not all_docs:
            raise ValueError("No content extracted from PDFs")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        final_documents = text_splitter.split_documents(all_docs)
        logger.info(f"Created {len(final_documents)} document chunks from {len(uploaded_files)} files")
        
        # Create vector store
        if st.session_state.embeddings is None:
            st.session_state.embeddings = get_embeddings()
        
        vectors = FAISS.from_documents(
            final_documents,
            st.session_state.embeddings
        )
        
        return vectors, len(uploaded_files), len(final_documents), len(all_docs)
        
    except Exception as e:
        raise
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_directory_pdfs(directory_path):
    """Process PDFs from a directory (existing functionality)"""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Documents directory not found: {directory_path}")
    
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError("No documents found in the directory")
    
    logger.info(f"Loaded {len(docs)} documents from directory")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    final_documents = text_splitter.split_documents(docs)
    logger.info(f"Created {len(final_documents)} document chunks")
    
    # Create vector store
    if st.session_state.embeddings is None:
        st.session_state.embeddings = get_embeddings()
    
    vectors = FAISS.from_documents(
        final_documents,
        st.session_state.embeddings
    )
    
    return vectors, len(docs), len(final_documents)


def initialize_llm_and_retriever():
    """Initialize and cache LLM and retriever to avoid recreating them"""
    # Initialize LLM if not cached
    if st.session_state.llm is None:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        logger.info("Initializing Groq LLM (caching for reuse)...")
        st.session_state.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=GROQ_MODEL,
            temperature=0.1,  # Lower temperature for more focused, concise responses
            timeout=20.0
        )
    
    # Initialize retriever if not cached
    if st.session_state.retriever is None:
        if st.session_state.vectors is None:
            raise ValueError("Vector store not initialized")
        
        logger.info("Creating retriever (caching for reuse)...")
        st.session_state.retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": NUM_RETRIEVED_DOCS}
        )


def get_answer(question: str, conversation_history: list = None) -> Optional[dict]:
    """Get answer to question using RAG with conversation context"""
    if st.session_state.vectors is None:
        return None
    
    try:
        logger.info(f"Processing question: {question[:50]}...")
        
        # Detect follow-up questions and get previous context
        previous_question = None
        previous_answer = None
        is_followup = False
        
        # Improved follow-up detection - understands natural language better
        question_lower = question.lower().strip()
        question_words = question_lower.split()
        
        if conversation_history and len(conversation_history) >= 2:
            # Get the last user question and assistant answer
            for i in range(len(conversation_history) - 1, -1, -1):
                if conversation_history[i]["role"] == "user":
                    previous_question = conversation_history[i]["content"]
                    break
            for i in range(len(conversation_history) - 1, -1, -1):
                if conversation_history[i]["role"] == "assistant":
                    previous_answer = conversation_history[i]["content"]
                    break
            
            # Smart follow-up detection - understands natural language patterns
            followup_patterns = [
                "elaborate", "tell me more", "explain more", "more details", "more about",
                "what about", "can you explain", "can you elaborate", "expand on",
                "go deeper", "dive deeper", "more information", "further details",
                "what else", "anything else", "also", "and", "how about", "what's more"
            ]
            
            # Check if question is very short (likely follow-up) or contains follow-up keywords
            is_short_followup = len(question_words) <= 4 and any(
                pattern in question_lower for pattern in ["more", "elaborate", "explain", "details", "about", "else"]
            )
            
            # Check if question contains follow-up phrases
            contains_followup_phrase = any(pattern in question_lower for pattern in followup_patterns)
            
            # Check if question is a pronoun reference (this, that, it, etc.)
            pronoun_followups = ["this", "that", "it", "they", "these", "those"]
            starts_with_pronoun = question_words and question_words[0] in pronoun_followups
            
            # If any condition is true and we have previous context, treat as follow-up
            if previous_question and (is_short_followup or contains_followup_phrase or starts_with_pronoun):
                is_followup = True
                logger.info(f"Detected follow-up question. Previous Q: {previous_question[:50]}")
        
        # Initialize embeddings if needed
        if st.session_state.embeddings is None:
            st.session_state.embeddings = get_embeddings()
        
        # Initialize LLM and retriever (cached)
        initialize_llm_and_retriever()
        
        # For follow-up questions, use the previous question for retrieval, but pass both to the LLM
        search_query = previous_question if (is_followup and previous_question) else question
        
        # Enhanced retrieval with relevance filtering
        logger.info("Retrieving relevant documents...")
        try:
            # Use invoke instead of deprecated get_relevant_documents
            retrieved_docs = st.session_state.retriever.invoke(search_query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Filter and rank documents by relevance to the actual question
            if retrieved_docs and not is_followup:
                question_words = set(question.lower().split())
                # Remove common stop words for better matching
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'where', 'when', 'who', 'how', 'my', 'your', 'his', 'her', 'their', 'our'}
                question_words = {w for w in question_words if w not in stop_words and len(w) > 2}
                
                if question_words:
                    doc_scores = []
                    for doc in retrieved_docs:
                        doc_text_lower = doc.page_content.lower()
                        # Count keyword matches
                        matches = sum(1 for word in question_words if word in doc_text_lower)
                        doc_scores.append((matches, doc))
                    
                    # Sort by relevance and take top documents
                    doc_scores.sort(reverse=True, key=lambda x: x[0])
                    retrieved_docs = [doc for _, doc in doc_scores[:NUM_RETRIEVED_DOCS] if _ > 0] or retrieved_docs[:NUM_RETRIEVED_DOCS]
            
            if not retrieved_docs:
                return {
                    "answer": "I could not find that information in the documents. Could you rephrase your question or try asking about something else?",
                    "context": [],
                    "time": 0.0
                }
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            raise ValueError(f"Failed to retrieve documents: {str(e)}")
        
        # Detect question type to determine answer depth
        question_lower = question.lower()
        needs_detail = any(word in question_lower for word in [
            "explain", "describe", "how", "why", "elaborate", "tell me about", 
            "what is", "what are", "details", "more about", "information about"
        ])
        is_simple_list = any(phrase in question_lower for phrase in [
            "what are my", "list my", "my skills", "my experience", "my education"
        ])
        
        # Create prompt template optimized for accuracy and adaptive answer depth
        if is_followup and previous_question and previous_answer:
            prompt = ChatPromptTemplate.from_template(
                """Answer the follow-up question using ONLY the provided context.

Previous: User asked "{previous_question}" and you answered: {previous_answer}
Current question: {question}
Context: {context}

Instructions:
- Use ONLY information from the context
- Provide additional details from context not in previous answer
- Match answer depth to question - if question asks for details, provide comprehensive answer
- If question is simple, keep answer brief
- If context doesn't contain the answer, say: "I could not find that information in the documents. Could you rephrase your question or try asking about something else?"

Answer:"""
            )
        else:
            if is_simple_list:
                # Simple list questions - concise answers
                prompt = ChatPromptTemplate.from_template(
                    """Answer the question concisely using ONLY the context below.

Context: {context}
Question: {question}

Instructions:
- Extract the requested information directly from the context
- Provide a brief, clear list or summary
- No categories, explanations, or structure unless asked
- If information is missing, say: "I could not find that information in the documents. Could you rephrase your question or try asking about something else?"

Answer:"""
                )
            elif needs_detail:
                # Questions needing detailed answers
                prompt = ChatPromptTemplate.from_template(
                    """Answer the question comprehensively using ONLY the context below.

Context: {context}
Question: {question}

Instructions:
- Provide a detailed, thorough answer based on the context
- Include relevant examples, specifics, and important details from the context
- Structure your answer clearly with paragraphs or sections as needed
- Be comprehensive but stay within the context - do not add external knowledge
- If information is missing, say: "I could not find that information in the documents. Could you rephrase your question or try asking about something else?"

Answer:"""
                )
            else:
                # Default - balanced answers
                prompt = ChatPromptTemplate.from_template(
                    """Answer the question using ONLY the context below. Match your answer depth to what the question requires.

Context: {context}
Question: {question}

Instructions:
- Extract information ONLY from the context
- If question asks for a list or simple fact, provide concise answer
- If question asks to explain or describe, provide detailed answer with examples from context
- Be precise and accurate - only use information from the context
- If information is missing, say: "I could not find that information in the documents. Could you rephrase your question or try asking about something else?"

Answer:"""
                )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
        
        # Create retrieval chain - handle both regular and follow-up questions
        if is_followup and previous_question:
            # For follow-ups, retrieve based on previous question but pass current question to LLM
            def get_context_with_followup(input_dict):
                return retrieved_docs  # Use already retrieved docs
            
            retrieval_chain = (
                {
                    "context": lambda x: retrieved_docs,  # Use pre-retrieved docs
                    "question": itemgetter("question"),
                    "previous_question": itemgetter("previous_question"),
                    "previous_answer": itemgetter("previous_answer"),
                }
                | document_chain
            )
        else:
            retrieval_chain = (
                {
                    "context": itemgetter("question") | st.session_state.retriever,
                    "question": itemgetter("question"),
                }
                | document_chain
            )
        
        # Get response with timeout
        logger.info("Getting answer from LLM...")
        start_time = time.time()
        try:
            # Prepare input for the chain
            chain_input = {"question": question}
            if is_followup and previous_question and previous_answer:
                chain_input["previous_question"] = previous_question
                chain_input["previous_answer"] = previous_answer
            
            response = retrieval_chain.invoke(chain_input)
            elapsed_time = time.time() - start_time
            logger.info(f"Got response in {elapsed_time:.2f} seconds")
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error in retrieval chain: {e}", exc_info=True)
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise ValueError(f"Request timed out after {elapsed_time:.1f} seconds. Please try again.")
            raise ValueError(f"Failed to get response: {str(e)}")
        
        return {
            "answer": response,
            "context": retrieved_docs,
            "time": elapsed_time
        }
        
    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise ValueError(f"Error processing question: {str(e)}")


# Hide deploy button with CSS
st.markdown("""
<style>
    .stDeployButton {
        display: none;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Main UI
st.title("Document Q&A System")
st.markdown("Upload PDF documents and ask questions using AI-powered semantic search")

# Validate API keys
is_valid, error_msg = validate_api_keys()
if not is_valid:
    st.error(f"Configuration Error: {error_msg}")
    st.info("Make sure your `.env` file contains:\n- GROQ_API_KEY\n- HUGGINGFACE_API_KEY")
    st.stop()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Current Status")
    if st.session_state.vectors:
        st.success("Vector store ready")
        if st.session_state.uploaded_files:
            st.caption(f"{len(st.session_state.uploaded_files)} file(s) loaded")
            for fname in st.session_state.uploaded_files[:3]:  # Show first 3
                st.caption(f"  • {fname}")
            if len(st.session_state.uploaded_files) > 3:
                st.caption(f"  ... and {len(st.session_state.uploaded_files) - 3} more")
        elif st.session_state.current_doc_set == "directory":
            st.caption("Directory documents loaded")
    else:
        st.info("No documents loaded")
    
    st.divider()
    
    st.subheader("Actions")
    if st.button("Clear All", help="Clear current documents and start fresh", use_container_width=True):
        st.session_state.vectors = None
        st.session_state.uploaded_files = []
        st.session_state.current_doc_set = None
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.llm = None
        st.rerun()
    
    st.divider()
    
    st.subheader("About")
    st.markdown("""
    **Document Q&A System**
    
    Powered by:
    - Groq (LLM)
    - HuggingFace (Embeddings)
    - FAISS (Vector Store)
    - LangChain (RAG Framework)
    """)

# Initialize embeddings
if st.session_state.embeddings is None:
    st.session_state.embeddings = get_embeddings()

# Document Processing Section
st.header("Document Processing")

st.markdown("### Upload PDF Documents")
st.markdown("Upload one or more PDF files to create a searchable knowledge base.")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="You can upload multiple PDF files at once. Each file will be processed and added to the knowledge base."
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    st.info(f"**{len(uploaded_files)} file(s) selected:** {', '.join(file_names[:3])}{'...' if len(file_names) > 3 else ''}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Process Uploaded PDFs", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing uploaded documents..."):
                    # Process uploaded files (no embedding method popup)
                    vectors, num_files, num_chunks, num_pages = process_uploaded_pdfs(uploaded_files)
                    
                    # Store in session state
                    st.session_state.vectors = vectors
                    st.session_state.uploaded_files = file_names
                    st.session_state.current_doc_set = "uploaded"
                    # Clear cached retriever to force recreation with new vectors
                    st.session_state.retriever = None
                    st.session_state.llm = None
                    
                    st.success(f"Successfully processed {num_files} PDF file(s) ({num_pages} pages) into {num_chunks} chunks!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                logger.error(f"Error processing uploaded files: {e}", exc_info=True)
    
    with col2:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.uploaded_files = []
            st.rerun()

# Show current document status
if st.session_state.vectors is not None:
    if st.session_state.uploaded_files:
        st.success(f"**Ready!** You can ask questions about: {', '.join(st.session_state.uploaded_files[:2])}{'...' if len(st.session_state.uploaded_files) > 2 else ''}")
    else:
        st.success("**Ready!** You can ask questions about the loaded documents.")

# Q&A Section
st.header("Ask Questions")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata_items = []
                        if 'page' in doc.metadata:
                            metadata_items.append(f"Page {doc.metadata['page']}")
                        if 'source' in doc.metadata:
                            source_name = Path(doc.metadata['source']).name
                            metadata_items.append(f"File: {source_name}")
                        if metadata_items:
                            st.caption(" | ".join(metadata_items))
                    st.divider()

# Question input
question = st.chat_input("Enter your question here...")

if question:
    if st.session_state.vectors is None:
        st.warning("Please upload and process documents first!")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer with progress indicator
        with st.chat_message("assistant"):
            try:
                # Use spinner for better UX
                with st.spinner("Searching documents and generating answer..."):
                    # Pass conversation history for context
                    result = get_answer(question, conversation_history=st.session_state.messages)
                
                if result:
                    st.write(result["answer"])
                    st.caption(f"Response time: {result['time']:.2f} seconds")
                    
                    # Display sources
                    with st.expander("Source Documents", expanded=False):
                        if result["context"]:
                            for i, doc in enumerate(result["context"], 1):
                                st.markdown(f"**Source {i}**")
                                st.write(doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    metadata_items = []
                                    if 'page' in doc.metadata:
                                        metadata_items.append(f"Page {doc.metadata['page']}")
                                    if 'source' in doc.metadata:
                                        source_name = Path(doc.metadata['source']).name
                                        metadata_items.append(f"File: {source_name}")
                                    if metadata_items:
                                        st.caption(" | ".join(metadata_items))
                                st.divider()
                        else:
                            st.info("No source documents found.")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("context", [])
                    })
                else:
                    st.error("Failed to get answer. Please try again.")
                    # Remove the user message if we failed
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                        st.session_state.messages.pop()
                        
            except ValueError as e:
                error_msg = str(e)
                st.error(f"{error_msg}")
                logger.error(f"Error: {error_msg}", exc_info=True)
                # Remove the user message if we failed
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                st.error(f"{error_msg}")
                logger.error(f"Unexpected error: {error_msg}", exc_info=True)
                # Remove the user message if we failed
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
