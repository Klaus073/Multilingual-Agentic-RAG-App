import streamlit as st
import os
from indexing import SemanticRAG
from retrieval import RAGRetriever
from chatbot import AgenticRAG
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Semantic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Center the main content */
    .main > div {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Style for chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Code block styling */
    pre {
        background-color: #1e1e1e;
        border-radius: 5px;
        padding: 1rem;
        overflow-x: auto;
    }
    
    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'semantic_rag' not in st.session_state:
    st.session_state.semantic_rag = None
if 'agentic_rag' not in st.session_state:
    st.session_state.agentic_rag = None

# Right sidebar for configuration and document upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI Configuration
    with st.expander("🔑 OpenAI Settings", expanded=True):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Enter your OpenAI API key"
        )
        
        model_name = st.text_input(
            "Model Name",
            value=os.getenv("DEFAULT_MODEL_NAME", "gpt-4o-mini"),
            help="Enter the OpenAI model name (e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo)"
        )
    
    # RAG Configuration
    with st.expander("🧠 RAG Settings", expanded=False):
        embedding_model = st.selectbox(
            "Embedding Model",
            ["bge-m3", "nomic-embed-text", "mxbai-embed-large"],
            index=["bge-m3", "nomic-embed-text", "mxbai-embed-large"].index(
                os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-m3")
            ),
            help="Select the Ollama embedding model"
        )
        
        breakpoint_threshold = st.slider(
            "Semantic Breakpoint Threshold",
            min_value=50,
            max_value=100,
            value=int(os.getenv("DEFAULT_BREAKPOINT_THRESHOLD", "70")),
            step=5,
            help="Lower = more chunks (more precise), Higher = fewer chunks (more context)"
        )
        
        buffer_size = st.number_input(
            "Buffer Size",
            min_value=1,
            max_value=5,
            value=int(os.getenv("DEFAULT_BUFFER_SIZE", "2")),
            help="Number of sentences to group when evaluating similarity"
        )
        
        top_k = st.number_input(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=int(os.getenv("DEFAULT_TOP_K", "10")),
            help="Number of search results to retrieve"
        )
    
    st.divider()
    
    # Document Upload Section
    st.header("📄 Document Upload")
    
    # Check if API key is provided
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key above to continue.")
    else:
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to index"
        )
        
        if uploaded_files and not st.session_state.indexed:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                try:
                    # Create progress container
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    # Initialize SemanticRAG
                    with st.spinner("Initializing semantic RAG system..."):
                        st.session_state.semantic_rag = SemanticRAG(
                            embedding_model=embedding_model,
                            breakpoint_threshold=breakpoint_threshold,
                            buffer_size=buffer_size,
                            use_gpu=False  # Set to False for broader compatibility
                        )
                    
                    # Create index from uploaded files
                    st.session_state.semantic_rag.create_index_from_files(
                        uploaded_files,
                        progress_callback=update_progress
                    )
                    
                    # Initialize retriever and agentic RAG
                    with st.spinner("Setting up chatbot..."):
                        rag_retriever = RAGRetriever(
                            rag_system=st.session_state.semantic_rag,
                            top_k=top_k
                        )
                        
                        st.session_state.agentic_rag = AgenticRAG(
                            retriever=rag_retriever,
                            openai_api_key=openai_api_key,
                            model_name=model_name
                        )
                    
                    st.session_state.indexed = True
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Successfully indexed {len(uploaded_files)} PDF(s)!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Error in document processing: {e}", exc_info=True)
        
        elif st.session_state.indexed:
            st.success(f"✅ Documents indexed and ready!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 New Documents", type="secondary", use_container_width=True):
                    st.session_state.indexed = False
                    st.session_state.messages = []
                    st.session_state.semantic_rag = None
                    st.session_state.agentic_rag = None
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                    st.session_state.messages = []
                    if st.session_state.agentic_rag:
                        st.session_state.agentic_rag.clear_memory()
                    st.rerun()

# Main chat interface
st.markdown("<h1 style='text-align: center;'>🤖 Semantic RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask questions about your uploaded documents using advanced AI</p>", unsafe_allow_html=True)

# Create a container for the chat interface
chat_container = st.container()

with chat_container:
    if not st.session_state.indexed:
        # Welcome message when no documents are indexed
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; margin: 2rem 0;'>
            <h2>👋 Welcome!</h2>
            <p style='font-size: 1.2rem; margin: 1rem 0;'>To get started:</p>
            <ol style='text-align: left; display: inline-block; font-size: 1.1rem;'>
                <li>Enter your OpenAI API key in the sidebar</li>
                <li>Upload one or more PDF documents</li>
                <li>Click "Process Documents" to index them</li>
                <li>Start asking questions!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat interface when documents are indexed
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Use markdown to properly render formatted content
                st.markdown(message["content"], unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.agentic_rag.ask(prompt)
                        
                        # Display the response with proper markdown rendering
                        message_placeholder.markdown(response, unsafe_allow_html=True)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        message_placeholder.error(error_msg)
                        logger.error(f"Error in chat: {e}", exc_info=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Powered by Semantic RAG with Ollama embeddings and OpenAI LLM</p>
        <p style='font-size: 0.9rem;'>Upload PDFs • Ask Questions • Get Intelligent Answers</p>
    </div>
    """,
    unsafe_allow_html=True
)