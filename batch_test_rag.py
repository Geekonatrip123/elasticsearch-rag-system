import json
import chromadb
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import time
import os

# --- Configuration ---
DB_PATH = "./chroma_db_advanced"
JSON_FILE = "questions.json"  # Change this to your JSON file name
OUTPUT_FILE = "rag_test_results.txt"

def setup_rag_system():
    """Initialize the RAG system (same as your query script)"""
    print("--- Setting up RAG system ---")
    
    # Configure models
    Settings.llm = Ollama(model="qwen3:4b", request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cpu"
    )
    
    # Load the persisted index
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection("advanced_docs_v1")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(
        persist_dir=DB_PATH, vector_store=vector_store
    )
    
    index = load_index_from_storage(storage_context)
    
    # Configure retriever and reranker
    base_retriever = index.as_retriever(similarity_top_k=12)
    retriever = AutoMergingRetriever(
        base_retriever,
        index.storage_context,
        verbose=False  # Set to False for batch processing
    )
    
    reranker = SentenceTransformerRerank(
        top_n=4,
        model="BAAI/bge-reranker-v2-m3",
        device="cpu"
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[reranker],
        streaming=False,  # Disable streaming for batch processing
    )
    
    print("--- RAG system ready ---")
    return query_engine

def load_questions(json_file):
    """Load questions from JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the text from each question
        questions = [item['text'] for item in data if 'text' in item]
        print(f"Loaded {len(questions)} questions from {json_file}")
        return questions
    
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file}")
        return []

def test_single_question(query_engine, question, question_num):
    """Test a single question and return results"""
    print(f"Testing question {question_num}: {question[:60]}...")
    
    start_time = time.time()
    try:
        response = query_engine.query(question)
        end_time = time.time()
        
        return {
            'question_num': question_num,
            'question': question,
            'response': str(response),
            'sources': [node.metadata.get('file_name', 'N/A') for node in response.source_nodes],
            'response_time': end_time - start_time,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        end_time = time.time()
        return {
            'question_num': question_num,
            'question': question,
            'response': None,
            'sources': [],
            'response_time': end_time - start_time,
            'success': False,
            'error': str(e)
        }

def save_results(results, output_file):
    """Save test results to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RAG SYSTEM TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        avg_time = sum(r['response_time'] for r in results) / total
        
        f.write(f"SUMMARY:\n")
        f.write(f"Total questions: {total}\n")
        f.write(f"Successful responses: {successful}\n")
        f.write(f"Failed responses: {total - successful}\n")
        f.write(f"Success rate: {successful/total*100:.1f}%\n")
        f.write(f"Average response time: {avg_time:.2f} seconds\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Question {result['question_num']}: {result['question']}\n")
            f.write(f"Time: {result['response_time']:.2f}s | Success: {result['success']}\n")
            
            if result['success']:
                f.write(f"Response: {result['response']}\n")
                f.write(f"Sources: {', '.join(result['sources'])}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            
            f.write("\n" + "="*80 + "\n\n")

def main():
    """Main testing function"""
    print("Starting batch RAG testing...")
    
    # Check if JSON file exists
    if not os.path.exists(JSON_FILE):
        print(f"Please save your JSON questions as '{JSON_FILE}' in the current directory")
        return
    
    # Load questions
    questions = load_questions(JSON_FILE)
    if not questions:
        return
    
    # Setup RAG system
    query_engine = setup_rag_system()
    
    # Test all questions
    results = []
    start_time = time.time()
    
    for i, question in enumerate(questions, 1):
        result = test_single_question(query_engine, question, i)
        results.append(result)
        
        # Print progress
        if i % 10 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(questions) - i)
            print(f"Progress: {i}/{len(questions)} | ETA: {eta/60:.1f} minutes")
    
    # Save results
    save_results(results, OUTPUT_FILE)
    
    total_time = time.time() - start_time
    print(f"\nTesting complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()