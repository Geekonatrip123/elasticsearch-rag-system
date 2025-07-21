# Advanced RAG System with Elasticsearch

Enterprise-grade RAG system using LlamaIndex + Elasticsearch for document search and question answering.

## 🏗️ Architecture Overview

```
Documents → LlamaIndex Processing → Elasticsearch (Docker) → RAG Pipeline → Qwen3 (Ollama)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Git
- CUDA-capable GPU (recommended)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd elasticsearch-rag-system
python -m venv environment
source environment/bin/activate  # On Windows: environment\Scripts\activate
pip install -r new_requirements.txt
```

### 2. Start Elasticsearch
```bash
docker-compose -f docker-compose-elasticsearch.yml up -d
```

Wait ~30 seconds for Elasticsearch to fully start, then verify:
```bash
curl http://localhost:9200
```

### 3. Start Ollama (Local LLM)
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model
ollama pull qwen3:4b

# Start Ollama server (if not auto-started)
ollama serve
```

### 4. Prepare Your Documents
```bash
# Create data directory and add your documents
mkdir -p data_large
# Copy your PDFs, DOCX, TXT files to data_large/
```

### 5. Build the Index
```bash
python 1_build_database_elasticsearch_fixed.py
```

### 6. Start Querying
```bash
python 2_query_system_elasticsearch_hierarchy.py
```

## 📁 Project Structure

```
elasticsearch-rag-system/
├── 1_build_database_elasticsearch_fixed.py  # Index builder
├── 2_query_system_elasticsearch_hierarchy.py # Query interface
├── 3_inspect_elasticsearch.py               # Debug/inspection tool
├── docker-compose-elasticsearch.yml         # Elasticsearch container
├── requirements.txt                         # Python dependencies
├── data_large/                              # Your documents (create this)
├── elasticsearch_storage_v2/                # Generated: Local storage
├── environment/                             # Generated: Python venv
└── README.md                               # This file
```

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file to customize settings:
```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=qwen3:4b

# Chunk Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Hardware Requirements
- **Minimum:** 8GB RAM, 4 CPU cores
- **Recommended:** 16GB RAM, 8 CPU cores, NVIDIA GPU
- **Storage:** 10GB+ free space (depends on document size)

## 📊 System Components

### 1. Document Processing
- **Framework:** LlamaIndex
- **Parser:** HierarchicalNodeParser (2048→512→128 tokens)
- **Formats:** PDF, DOCX, TXT, MD, JSON
- **Tokenizer:** tiktoken (GPT-compatible)

### 2. Vector Storage
- **Engine:** Elasticsearch 8.11.0 (Docker)
- **Index:** Hybrid search (vector + text)
- **Embeddings:** sentence-transformers/all-mpnet-base-v2
- **Dimensions:** 768

### 3. Retrieval System
- **Base Retriever:** Elasticsearch hybrid search
- **Enhancement:** AutoMergingRetriever (hierarchy-aware)
- **Reranking:** BGE reranker (BAAI/bge-reranker-v2-m3)
- **Top-K:** 12 → 4 after reranking

### 4. Generation
- **LLM:** Qwen3:4b via Ollama
- **Response:** Streaming with thinking tags
- **Context:** Retrieved chunks + metadata

## 🐳 Docker Services

### Elasticsearch Container
```yaml
# docker-compose-elasticsearch.yml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
```

### Container Management
```bash
# Start services
docker-compose -f docker-compose-elasticsearch.yml up -d

# Check status
docker-compose -f docker-compose-elasticsearch.yml ps

# View logs
docker-compose -f docker-compose-elasticsearch.yml logs elasticsearch

# Stop services
docker-compose -f docker-compose-elasticsearch.yml down

# Clean up (removes data)
docker-compose -f docker-compose-elasticsearch.yml down -v
```

## 🔍 Usage Examples

### Basic Query
```bash
python 2_query_system_elasticsearch_hierarchy.py
```
```
Question: explain lottery scheduling
🤖 Response: Lottery scheduling is a probabilistic scheduling algorithm...
```

### Inspect System
```bash
python 3_inspect_elasticsearch.py
```
Shows index statistics, document counts, and sample data.

### Advanced Usage
```python
# Custom query with filters
from elasticsearch_client import ElasticsearchClient

client = ElasticsearchClient()
results = client.search(
    query="machine learning", 
    filters={"metadata.file_type": "pdf"},
    size=10
)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Elasticsearch Won't Start
```bash
# Check if port is in use
netstat -tulpn | grep 9200

# Check Docker logs
docker logs elasticsearch-rag

# Restart with clean state
docker-compose down -v && docker-compose up -d
```

#### 2. CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python 1_build_database_elasticsearch_fixed.py
```

#### 3. Out of Memory
```bash
# Reduce batch size in build script
# Edit embedding batch_size from 32 to 16
python 1_build_database_elasticsearch_fixed.py
```

#### 4. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check model availability
ollama list
```

### Performance Tuning

#### For Large Document Collections
```bash
# Increase Elasticsearch memory
docker-compose down
# Edit docker-compose-elasticsearch.yml:
# ES_JAVA_OPTS=-Xms2g -Xmx4g
docker-compose up -d
```

#### For Slow Queries
```bash
# Reduce retrieval size
# Edit query script: similarity_top_k=6 instead of 12
# Disable reranking temporarily for testing
```

## 📈 Performance Metrics

### Typical Performance
- **Index Build:** ~500 docs/minute
- **Query Response:** <2 seconds
- **Memory Usage:** ~2GB during indexing, ~500MB during querying
- **Storage:** ~50MB per 1000 document chunks

### Benchmarking
```bash
# Time index building
time python 1_build_database_elasticsearch_fixed.py

# Check Elasticsearch stats
curl http://localhost:9200/advanced_docs_elasticsearch_v2/_stats
```

## 🔒 Security Notes

### Development Mode
- Elasticsearch runs without authentication
- All data is stored locally
- No encryption in transit

### Production Deployment
- Enable Elasticsearch security features
- Use HTTPS for all communications
- Implement proper authentication
- Regular security updates

## 🛠️ Development

### Adding New Document Types
1. Extend `DocumentProcessor` class
2. Add file extension to `supported_extensions`
3. Implement extraction method
4. Test with sample files

### Customizing Retrieval
1. Modify `similarity_top_k` in query script
2. Adjust chunk sizes in build script
3. Change reranking model
4. Experiment with different embedding models

### API Integration
```python
# Example Flask wrapper
from flask import Flask, request, jsonify
from elk_rag_system import ELKRAGSystem

app = Flask(__name__)
rag_system = ELKRAGSystem()

@app.route('/query', methods=['POST'])
def query():
    question = request.json['question']
    result = rag_system.query(question)
    return jsonify(result)
```

## 📚 Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Ollama Documentation](https://ollama.ai/docs)
- [sentence-transformers](https://www.sbert.net/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Search existing GitHub issues
3. Create new issue with system details and error logs
