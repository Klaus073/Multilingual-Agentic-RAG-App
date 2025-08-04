# Semantic RAG Chatbot

A powerful document Q&A system that combines semantic chunking, vector search, and OpenAI's language models to answer questions about your PDF documents.

## Features

- **Semantic Document Processing**: Intelligently chunks documents based on semantic similarity
- **Vector Search**: Fast and accurate retrieval using FAISS
- **Agentic RAG**: Smart routing between greetings and document queries
- **Conversational Memory**: Maintains context across questions
- **User-Friendly UI**: Clean Streamlit interface with progress tracking

## Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running locally (for embeddings)
3. **OpenAI API Key**

## Installation

1. Clone or download the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

4. Pull the embedding model:
```bash
ollama pull bge-m3
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. In the web interface:
   - Enter your OpenAI API key in the sidebar
   - Configure the model name (default: gpt-4o-mini)
   - Upload one or more PDF files
   - Click "Process Documents" and wait for indexing to complete
   - Start asking questions!

## Configuration Options

### Embedding Models
- `bge-m3` (default) - 1024 dimensions
- `nomic-embed-text` - 768 dimensions
- `mxbai-embed-large` - 1024 dimensions

### Semantic Chunking Parameters
- **Breakpoint Threshold** (50-100): Lower values create more, smaller chunks
- **Buffer Size** (1-5): Number of sentences to group when evaluating similarity

### OpenAI Models
- `gpt-4o-mini`  - Fast and cost-effective
- `gpt-4` - Most capable
- `gpt-3.5-turbo` - Good balance of speed and capability

## Architecture

The system consists of four main modules:

1. **indexing.py**: Handles PDF processing and semantic chunking
2. **retrieval.py**: Manages vector search and basic RAG chains
3. **chatbot.py**: Implements the agentic workflow with query classification
4. **streamlit_app.py**: Provides the web interface

## Troubleshooting

### Ollama Connection Error
Make sure Ollama is running:
```bash
ollama serve
```

### GPU Support
To use GPU acceleration with FAISS:
1. Install `faiss-gpu` instead of `faiss-cpu`
2. Set `use_gpu=True` in the configuration

### Memory Issues
For large documents, you may need to:
- Reduce the number of search results (`top_k`)
- Increase the breakpoint threshold for fewer chunks
- Process documents in smaller batches

## License

This project is open source and available under the MIT License.