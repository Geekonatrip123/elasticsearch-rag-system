# 3_inspect_elasticsearch.py
import os
from elasticsearch import Elasticsearch
from llama_index.core import StorageContext

# --- 0. Define Constants ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch"
ES_STORAGE_DIR = "./elasticsearch_storage"

def inspect_elasticsearch_index():
    """Inspect the Elasticsearch index"""
    print("=== Inspecting Elasticsearch Index ===")
    
    try:
        es_client = Elasticsearch([ES_ENDPOINT])
        es_info = es_client.info()
        print(f"Connected to Elasticsearch {es_info['version']['number']}")
        
        if not es_client.indices.exists(index=INDEX_NAME):
            print(f"❌ Index '{INDEX_NAME}' does not exist!")
            return
        
        # Get index stats
        stats = es_client.indices.stats(index=INDEX_NAME)
        doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']
        store_size = stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes']
        
        print(f"\n📊 Index Statistics:")
        print(f"  Index name: {INDEX_NAME}")
        print(f"  Document count: {doc_count}")
        print(f"  Store size: {store_size / (1024*1024):.2f} MB")
        
        # Get index mapping
        mapping = es_client.indices.get_mapping(index=INDEX_NAME)
        properties = mapping[INDEX_NAME]['mappings']['properties']
        
        print(f"\n🔍 Fields in index:")
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'object')
            if field_type == 'dense_vector':
                dims = field_info.get('dims', 'unknown')
                print(f"  - {field_name}: {field_type} (dims: {dims})")
            else:
                print(f"  - {field_name}: {field_type}")
        
        # Sample some documents
        print(f"\n📄 Sample documents:")
        search_result = es_client.search(index=INDEX_NAME, size=2)
        
        for i, hit in enumerate(search_result['hits']['hits']):
            source = hit['_source']
            content_preview = source.get('content', '')[:150] + "..."
            print(f"\n  Document {i+1}:")
            print(f"    Content: {content_preview}")
            if 'metadata' in source:
                print(f"    File: {source['metadata'].get('file_name', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error inspecting Elasticsearch: {e}")

def inspect_local_storage():
    """Inspect the local storage (docstore)"""
    print("\n=== Inspecting Local Storage ===")
    
    if not os.path.exists(ES_STORAGE_DIR):
        print(f"❌ Local storage directory not found: {ES_STORAGE_DIR}")
        return
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=ES_STORAGE_DIR)
        docstore = storage_context.docstore
        
        all_nodes = list(docstore.docs.values())
        print(f"📁 Docstore contains {len(all_nodes)} nodes")
        
        # Count node types
        parent_nodes = [n for n in all_nodes if n.child_nodes]
        leaf_nodes = [n for n in all_nodes if not n.child_nodes]
        
        print(f"  - Parent nodes: {len(parent_nodes)}")
        print(f"  - Leaf nodes: {len(leaf_nodes)}")
        
        # Show hierarchy info
        if parent_nodes:
            print(f"  - Average children per parent: {sum(len(n.child_nodes) for n in parent_nodes) / len(parent_nodes):.1f}")
        
    except Exception as e:
        print(f"❌ Error inspecting local storage: {e}")

if __name__ == "__main__":
    inspect_elasticsearch_index()
    inspect_local_storage()