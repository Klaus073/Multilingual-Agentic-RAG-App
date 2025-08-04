import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext
import faiss
import time

class SemanticRAG:
    def __init__(self, embedding_model="bge-m3", breakpoint_threshold=95, buffer_size=1, use_gpu=True):
        """
        Initialize the Semantic Splitter RAG system
        
        Args:
            embedding_model: Ollama embedding model name
            breakpoint_threshold: Percentile threshold for semantic breaks (95=fewer chunks, 80=more chunks)
            buffer_size: Number of sentences to group when evaluating similarity (1=individual sentences)
            use_gpu: Whether to use GPU acceleration for FAISS
        """
        self.embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.buffer_size = buffer_size
        self.use_gpu = use_gpu
        self.index = None
        
        # Check GPU availability
        if use_gpu:
            try:
                self.gpu_available = faiss.get_num_gpus() > 0
                if self.gpu_available:
                    print(f"✓ GPU detected: {faiss.get_num_gpus()} GPU(s) available")
                else:
                    print("No GPU detected, using CPU")
                    self.use_gpu = False
            except:
                print("GPU not available, using CPU")
                self.use_gpu = False
        
        # Disable LLM (we only want retrieval)
        Settings.llm = None
        print("✓ LLM disabled - retrieval only mode")
        
        # Setup Ollama embeddings
        try:
            print(f"Initializing embedding model: {embedding_model}")
            self.embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url="http://localhost:11434"
            )
            Settings.embed_model = self.embed_model
            print("✓ Ollama embedding model initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            raise
        
        # Setup semantic splitter node parser
        print(f"Creating SemanticSplitterNodeParser...")
        print(f"  - Breakpoint threshold: {breakpoint_threshold}% (lower = more chunks)")
        print(f"  - Buffer size: {buffer_size} sentence(s) per evaluation")
        
        self.node_parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embed_model,
            breakpoint_percentile_threshold=breakpoint_threshold,
            buffer_size=buffer_size,
            original_text_metadata_key="original_text"
        )
        print("✓ SemanticSplitterNodeParser created successfully")
    
    def _get_embedding_dimension(self):
        """Get embedding dimension for FAISS index"""
        try:
            print("Detecting embedding dimension...")
            sample_embedding = self.embed_model.get_text_embedding("test")
            dimension = len(sample_embedding)
            print(f"✓ Detected dimension: {dimension}")
            return dimension
        except Exception as e:
            print(f"Could not detect dimension: {e}")
            # Fallback for common models
            fallback_dims = {
                "bge-m3": 1024,
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024
            }
            dimension = fallback_dims.get(self.embedding_model, 768)
            print(f"Using fallback dimension: {dimension}")
            return dimension
    
    def create_index_from_files(self, pdf_files, progress_callback=None):
        """
        Create index from uploaded PDF files with progress updates
        
        Args:
            pdf_files: List of uploaded file objects (Streamlit UploadedFile objects)
            progress_callback: Optional callback function for progress updates
        """
        print(f"Loading {len(pdf_files)} PDF files")
        start_time = time.time()
        
        if progress_callback:
            progress_callback(0.1, "Starting document processing...")
        
        documents = []
        
        # Process each PDF individually with proper page tracking
        try:
            import fitz  # pymupdf
            print("✓ Using pymupdf for PDF extraction")
            
            for file_idx, pdf_file in enumerate(pdf_files):
                if progress_callback:
                    progress_callback(
                        0.1 + (0.3 * file_idx / len(pdf_files)), 
                        f"Processing PDF {file_idx + 1}/{len(pdf_files)}: {pdf_file.name}"
                    )
                
                print(f"\n📖 Processing PDF {file_idx + 1}/{len(pdf_files)}: {pdf_file.name}")
                
                # Read file content from Streamlit UploadedFile
                file_content = pdf_file.read()
                doc = fitz.open(stream=file_content, filetype="pdf")
                file_page_count = len(doc)
                print(f"   📊 Total pages in this file: {file_page_count}")
                
                for page_num in range(file_page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    
                    if text.strip():
                        from llama_index.core.schema import Document
                        page_doc = Document(text=text)
                        
                        # CRITICAL: Page number relative to THIS file only
                        local_page_num = page_num + 1
                        
                        page_doc.metadata = {
                            'file_name': pdf_file.name,
                            'filename': pdf_file.name,
                            'page_number': local_page_num,  # 1-based within this file
                            'page_label': str(local_page_num),
                            'total_pages_in_file': file_page_count,
                            'file_index': file_idx
                        }
                        
                        documents.append(page_doc)
                        print(f"   ✓ Page {local_page_num}/{file_page_count}")
                
                doc.close()
                print(f"   📚 Loaded {file_page_count} pages from {pdf_file.name}")
                
        except ImportError:
            print("⚠️  pymupdf not available - install with: pip install pymupdf")
            raise Exception("pymupdf is required for PDF processing")
        
        if progress_callback:
            progress_callback(0.4, f"Creating semantic chunks from {len(documents)} pages...")
        
        # Parse semantic chunks
        print(f"\n🧠 Creating semantic chunks from {len(documents)} pages...")
        nodes = self.node_parser.build_semantic_nodes_from_documents(documents, show_progress=True)
        
        # Create comprehensive mapping
        doc_id_to_info = {}
        for doc in documents:
            doc_id_to_info[doc.doc_id] = {
                'filename': doc.metadata.get('filename', 'unknown'),
                'page_number': doc.metadata.get('page_number', 1),
                'total_pages_in_file': doc.metadata.get('total_pages_in_file', 1)
            }
        
        # Map chunks to pages
        mapped_chunks = 0
        for i, node in enumerate(nodes):
            if hasattr(node, 'ref_doc_id') and node.ref_doc_id in doc_id_to_info:
                source_info = doc_id_to_info[node.ref_doc_id]
                node.metadata.update({
                    'filename': source_info['filename'],
                    'page_number': source_info['page_number'],  # REAL page number
                    'mapping_status': 'accurate'
                })
                mapped_chunks += 1
            else:
                node.metadata.update({
                    'filename': 'unknown',
                    'page_number': 'unknown',
                    'mapping_status': 'failed'
                })
            
            node.metadata['chunk_id'] = f"semantic_chunk_{i+1}"
        
        print(f"✅ Successfully mapped: {mapped_chunks}/{len(nodes)} chunks")
        
        if progress_callback:
            progress_callback(0.7, "Creating vector index...")
        
        # Create index
        dimension = self._get_embedding_dimension()
        faiss_index = faiss.IndexFlatIP(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        if progress_callback:
            progress_callback(0.8, "Generating embeddings...")
        
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        
        if progress_callback:
            progress_callback(1.0, "✅ Index created successfully!")
        
        print("✅ Semantic RAG index with REAL page numbers created!")
        return self.index
    
    def search(self, query, top_k=3):
        """
        Search the index using semantic chunk retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with semantic chunks
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        print(f"🔍 Searching: '{query}'")
        
        # Create retriever
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Format results
        results = []
        for i, node in enumerate(nodes):
            result = {
                'rank': i + 1,
                'score': getattr(node, 'score', None),
                'filename': node.metadata.get('filename', 'Unknown'),
                'page_number': node.metadata.get('page_number', 'Unknown'),
                'chunk_id': node.metadata.get('chunk_id', f'chunk_{i+1}'),
                'chunk_type': node.metadata.get('chunk_type', 'semantic'),
                'text': node.text,
                'text_length': len(node.text),
                'original_text': node.metadata.get('original_text', node.text)
            }
            results.append(result)
        
        print(f"✓ Found {len(results)} semantic chunks")
        return results