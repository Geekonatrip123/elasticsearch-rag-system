# Advanced RAG System with Hierarchical Document Processing

A sophisticated Retrieval-Augmented Generation (RAG) system built with LlamaIndex, ChromaDB, and Ollama that provides intelligent document querying with advanced features like hierarchical chunking, auto-merging retrieval, and re-ranking.

## 🚀 Features

- **Hierarchical Document Processing**: Implements advanced chunking strategies for better context preservation
- **Auto-Merging Retrieval**: Intelligently merges related document chunks for comprehensive answers
- **Advanced Re-ranking**: Uses state-of-the-art reranking models to improve relevance
- **Real-time Streaming**: Displays model thinking process and answers in real-time
- **GPU Acceleration**: Leverages CUDA for fast embeddings and reranking
- **Persistent Storage**: Uses ChromaDB for efficient vector storage and retrieval

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Hierarchical   │───▶│   ChromaDB      │
│   (PDF, etc.)   │    │   Processing    │    │ Vector Store    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query UI      │◄───│ Auto-Merging    │◄───│   Retriever     │
│  (Terminal)     │    │   + Reranking   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Ollama installed and running
- At least 8GB RAM (16GB+ recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SME
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv environment
   # On Windows
   .\environment\Scripts\activate
   # On Linux/Mac
   source environment/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   - Download and install Ollama from [https://ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull qwen3:4b
     ```

## 📁 Project Structure

```
SME/
├── 1_build_database_advanced.py    # Database creation and document processing
├── 2_query_system_advanced.py      # Interactive query interface
├── 3_inspect_hierarchy.py          # Utility to inspect document hierarchy
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data_large/                     # Directory for input documents
│   └── *.pdf                       # PDF documents to process
├── chroma_db_advanced/             # ChromaDB storage (auto-generated)
│   ├── chroma.sqlite3
│   ├── docstore.json
│   ├── graph_store.json
│   ├── image__vector_store.json
│   ├── index_store.json
│   └── [collection-id]/
└── environment/                    # Virtual environment (not in git)
```

## 🚀 Quick Start

### Step 1: Prepare Your Documents
Place your PDF documents in the `data_large/` directory.

### Step 2: Build the Database
```bash
python 1_build_database_advanced.py
```
This script will:
- Process your documents with hierarchical chunking
- Generate embeddings using HuggingFace models
- Store everything in ChromaDB for fast retrieval

### Step 3: Start Querying
```bash
python 2_query_system_advanced.py
```
This launches an interactive terminal interface where you can:
- Ask questions about your documents
- See real-time thinking process from the AI
- View source references and confidence scores

### Step 4: (Optional) Inspect Document Hierarchy
```bash
python 3_inspect_hierarchy.py
```
Use this to understand how your documents were processed and chunked.

## 🎯 Usage Examples

Once you run `2_query_system_advanced.py`, you can ask questions like:

```
Question: What are the main concepts of operating systems?

🤖 Response: 
🤔 Thinking: Let me search through the operating systems documentation to find the main concepts...

💭 Final Answer: Based on the documentation, the main concepts of operating systems include:

1. **Virtualization**: The OS provides abstractions of physical resources
2. **Concurrency**: Managing multiple processes simultaneously
3. **Persistence**: Storing data reliably on storage devices
...
```

## ⚙️ Configuration

### Models Used
- **LLM**: Qwen3:4b (via Ollama)
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **Reranker**: BAAI/bge-reranker-v2-m3

### Key Parameters
```python
# In 1_build_database_advanced.py
CHUNK_SIZE = 512           # Base chunk size
CHUNK_OVERLAP = 128        # Overlap between chunks

# In 2_query_system_advanced.py
SIMILARITY_TOP_K = 12      # Initial retrieval count
RERANK_TOP_N = 4          # Final reranked results
```

## 🔧 Customization

### Adding New Document Types
Modify `1_build_database_advanced.py` to support additional file formats:
```python
# Add new loaders in the document loading section
from llama_index.readers.file import DocxReader
# ... add your loader logic
```

### Changing Models
Update the model configurations in both scripts:
```python
# For different LLM
Settings.llm = Ollama(model="your-model:tag")

# For different embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="your-embedding-model"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Check CUDA availability
   python test_cuda.py
   
   # Reduce batch sizes or switch to CPU
   device="cpu"  # in model configurations
   ```

2. **Ollama Connection Issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check if model is available
   ollama list
   ```

3. **ChromaDB Permission Issues**
   ```bash
   # Delete and rebuild database
   rm -rf chroma_db_advanced/
   python 1_build_database_advanced.py
   ```

### Performance Optimization

- **GPU Memory**: Reduce `similarity_top_k` if running out of memory
- **Speed**: Use smaller embedding models for faster processing
- **Quality**: Increase `chunk_overlap` for better context preservation

## 📊 Testing

Run the included test utilities:

```bash
# Test CUDA setup
python test_cuda.py

# Test context window handling
python test_context_window.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace](https://huggingface.co/) for transformer models

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Note**: This project requires significant computational resources. For best performance, use a machine with a CUDA-compatible GPU and sufficient RAM.
