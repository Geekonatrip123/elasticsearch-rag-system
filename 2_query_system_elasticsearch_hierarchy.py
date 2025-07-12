# 2_query_system_elasticsearch_hierarchy.py
import os
from elasticsearch import Elasticsearch
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

# --- 0. Define Constants ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"
ES_STORAGE_DIR = "./elasticsearch_storage_v2"

# --- 1. CONFIGURE MODELS ---
print("--- Configuring models ---")
Settings.llm = Ollama(model="qwen3:4b", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

# --- 2. LOAD WITH PROPER HIERARCHY SUPPORT ---
print(f"--- Loading data from Elasticsearch and local storage ---")

try:
    # Connect to Elasticsearch
    es_client = Elasticsearch([ES_ENDPOINT])
    es_info = es_client.info()
    print(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
    
    # Check if index exists
    if not es_client.indices.exists(index=INDEX_NAME):
        print(f"❌ Index '{INDEX_NAME}' does not exist!")
        print("Please run '1_build_database_elasticsearch_fixed.py' first.")
        exit()
    
    # Create Elasticsearch vector store
    vector_store = ElasticsearchStore(
        index_name=INDEX_NAME,
        es_url=ES_ENDPOINT,
        vector_field="embedding",
        text_field="content"
    )
    
    # Load storage context with hierarchy
    if not os.path.exists(ES_STORAGE_DIR):
        print(f"❌ Storage directory not found at '{ES_STORAGE_DIR}'.")
        print("Please run '1_build_database_elasticsearch_fixed.py' first.")
        exit()
    
    storage_context = StorageContext.from_defaults(
        persist_dir=ES_STORAGE_DIR,
        vector_store=vector_store
    )
    print(f"✅ Loaded storage context with {len(storage_context.docstore.docs)} nodes")
    
    # Create index from storage context (maintains hierarchy)
    print("--- Creating index with hierarchy support ---")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

except Exception as e:
    print(f"❌ Error loading index: {e}")
    exit()

# --- 3. CONFIGURE ADVANCED RETRIEVER WITH HIERARCHY ---
print("--- Configuring AutoMergingRetriever with hierarchy ---")

try:
    # Base retriever
    base_retriever = index.as_retriever(similarity_top_k=12)
    
    # AutoMerging retriever (this needs the hierarchy)
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,  # Pass storage_context directly
        verbose=True
    )
    
    # Reranker
    reranker = SentenceTransformerRerank(
        top_n=4,
        model="BAAI/bge-reranker-v2-m3",
        device="cuda"
    )
    
    # Query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[reranker],
        streaming=True,
    )
    
    print("✅ AutoMergingRetriever configured successfully")

except Exception as e:
    print(f"❌ Error configuring AutoMergingRetriever: {e}")
    print("Falling back to simple retriever...")
    
    # Fallback to simple retriever
    base_retriever = index.as_retriever(similarity_top_k=10)
    reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-v2-m3", device="cuda")
    query_engine = RetrieverQueryEngine.from_args(base_retriever, node_postprocessors=[reranker], streaming=True)

def print_qwen3_response_stream(response_stream):
    """Custom function to handle Qwen3's thinking tags while streaming"""
    full_text = ""
    in_think_tag = False
    buffer = ""
    
    print("🤖 Response: ", end="", flush=True)
    
    for text in response_stream.response_gen:
        full_text += text
        buffer += text
        
        while buffer:
            if "<think>" in buffer and not in_think_tag:
                think_start = buffer.find("<think>")
                if think_start > 0:
                    print(buffer[:think_start], end="", flush=True)
                
                in_think_tag = True
                print("\n\n🤔 Thinking: ", end="", flush=True)
                buffer = buffer[think_start + 7:]
                continue
            
            elif "</think>" in buffer and in_think_tag:
                think_end = buffer.find("</think>")
                if think_end > 0:
                    print(buffer[:think_end], end="", flush=True)
                
                in_think_tag = False
                print("\n\n💭 Final Answer: ", end="", flush=True)
                buffer = buffer[think_end + 8:]
                continue
            
            else:
                if not ("<think>" in buffer or "</think>" in buffer):
                    print(buffer, end="", flush=True)
                    buffer = ""
                else:
                    break
    
    if buffer:
        print(buffer, end="", flush=True)
    
    print()
    return full_text

# --- 4. ASK QUESTIONS ---
print("\n🚀 Ready to Query with Elasticsearch + AutoMerging!")
print("Ask questions about your documents. Type 'exit' or press Ctrl+C to quit.")

try:
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break

        print(f"\n🔍 Searching with hierarchy merging: '{question}'")
        
        try:
            response = query_engine.query(question)
            print_qwen3_response_stream(response)

            print("\n--- Source Nodes (Post-AutoMerging & Re-ranking) ---")
            for i, node in enumerate(response.source_nodes):
                print(f"Source {i+1} (Score: {node.score:.4f}):")
                cleaned_text = ' '.join(node.get_content().split())
                print(f"  -> File: {node.metadata.get('file_name', 'N/A')}")
                print(f"  -> Node Type: {'Merged' if len(cleaned_text) > 1000 else 'Leaf'}")
                print(f"  -> Content Snippet: \"{cleaned_text[:250]}...\"\n")
            print("--------------------------------------")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("Continuing with next question...")

except KeyboardInterrupt:
    print("\nExiting gracefully. Goodbye!")