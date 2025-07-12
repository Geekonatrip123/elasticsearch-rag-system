import os
from elasticsearch import Elasticsearch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 0. Define Constants ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"  # New index name to avoid conflicts
ES_STORAGE_DIR = "./elasticsearch_storage_v2"  # New storage directory
DATA_PATH = "./data_large"

# --- 1. CONFIGURE MODELS ---
print("--- Configuring models ---")
Settings.llm = Ollama(model="qwen3:8b", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cuda"
)

# --- 2. CHECK ELASTICSEARCH CONNECTION ---
print(f"--- Connecting to Elasticsearch at {ES_ENDPOINT} ---")
try:
    es_client = Elasticsearch([ES_ENDPOINT])
    es_info = es_client.info()
    print(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
    
    # Delete existing index if it exists (fresh start)
    if es_client.indices.exists(index=INDEX_NAME):
        es_client.indices.delete(index=INDEX_NAME)
        print(f"🗑️ Deleted existing index: {INDEX_NAME}")
        
except Exception as e:
    print(f"❌ Failed to connect to Elasticsearch: {e}")
    exit()

# --- 3. INITIALIZE ELASTICSEARCH STORAGE ---
print(f"--- Setting up Elasticsearch vector store ---")
vector_store = ElasticsearchStore(
    index_name=INDEX_NAME,
    es_url=ES_ENDPOINT,
    vector_field="embedding",
    text_field="content"
)

# Create fresh storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 4. LOAD DOCUMENTS ---
print(f"--- Loading documents from '{DATA_PATH}' ---")
if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)
    print(f"Created '{DATA_PATH}' directory. Please add your documents and run again.")
    exit()

documents = SimpleDirectoryReader(DATA_PATH).load_data()
print(f"Loaded {len(documents)} document(s).")

# --- 5. PARSE WITH HIERARCHICAL NODE PARSER ---
print("--- Parsing documents with HierarchicalNodeParser ---")
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],
    chunk_overlap=20  # Add some overlap for better continuity
)
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
print(f"Generated {len(nodes)} total nodes and {len(leaf_nodes)} leaf nodes for embedding.")

# --- 6. CRITICAL FIX: Add ALL nodes to docstore FIRST ---
print(f"--- Adding ALL {len(nodes)} nodes to docstore (including parents) ---")
storage_context.docstore.add_documents(nodes)

# Verify all nodes are in docstore
docstore_count = len(storage_context.docstore.docs)
print(f"✅ Docstore now contains {docstore_count} nodes")

# --- 7. BUILD INDEX FROM LEAF NODES ---
print(f"--- Building vector index from {len(leaf_nodes)} leaf nodes... ---")

# Build index with leaf nodes only (for embeddings)
vector_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    show_progress=True,
)

# --- 8. VERIFY HIERARCHY RELATIONSHIPS ---
print("--- Verifying hierarchy relationships ---")
hierarchy_ok = True
missing_parents = []

for leaf_node in leaf_nodes:
    if hasattr(leaf_node, 'parent_node') and leaf_node.parent_node:
        parent_id = leaf_node.parent_node.node_id
        if parent_id not in storage_context.docstore.docs:
            missing_parents.append(parent_id)
            hierarchy_ok = False

if hierarchy_ok:
    print("✅ All parent-child relationships verified successfully")
else:
    print(f"⚠️ Warning: {len(missing_parents)} parent nodes missing from docstore")
    print("This may cause AutoMergingRetriever issues")

# --- 9. PERSIST STORAGE ---
print(f"--- Persisting storage to '{ES_STORAGE_DIR}' ---")
if os.path.exists(ES_STORAGE_DIR):
    import shutil
    shutil.rmtree(ES_STORAGE_DIR)
    print(f"🗑️ Removed existing storage directory")

storage_context.persist(persist_dir=ES_STORAGE_DIR)

# --- 10. FINAL VERIFICATION ---
print("--- Final verification ---")
try:
    # Test loading the storage context
    test_storage = StorageContext.from_defaults(
        persist_dir=ES_STORAGE_DIR,
        vector_store=vector_store
    )
    test_docstore_count = len(test_storage.docstore.docs)
    
    # Test creating index
    test_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=test_storage
    )
    
    print(f"✅ Verification successful!")
    print(f"   - Docstore: {test_docstore_count} nodes")
    print(f"   - Index: Ready for querying")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")

print("\n--- Ingestion Complete ---")
print(f"✅ Successfully built index with Elasticsearch vector store")
print(f"📁 Docstore saved to: {ES_STORAGE_DIR}")
print(f"🔍 Vector embeddings stored in Elasticsearch index: {INDEX_NAME}")
print(f"🏗️ Hierarchy preserved for AutoMergingRetriever")