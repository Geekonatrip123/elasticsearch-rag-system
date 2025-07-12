# 3_inspect_hierarchy.py (DEFINITIVE SCRIPT v3)
# A standalone script to visualize the node hierarchy created by HierarchicalNodeParser.

import os
from llama_index.core import StorageContext

# --- 0. Define Constants ---
DB_PATH = "./chroma_db_advanced"

def print_node_hierarchy(nodes, docstore, level=0):
    """
    Recursively prints the hierarchy of nodes.
    
    Args:
        nodes (list): A list of nodes to print.
        docstore (SimpleDocumentStore): The document store containing all nodes.
        level (int): The current indentation level for printing.
    """
    for node in nodes:
        # Indentation for visual hierarchy
        indent = "  " * level
        
        # Get a snippet of the node's content for display
        content_snippet = ' '.join(node.get_content().split())[:100] + "..."
        
        # Print node information
        print(f"{indent}* Node ID: {node.node_id}")
        print(f"{indent}  - Type: {'Parent' if node.child_nodes else 'Leaf'}")
        print(f"{indent}  - Snippet: \"{content_snippet}\"")
        
        # If the node has children, find them in the docstore and recurse
        if node.child_nodes:
            # FIX #1: Get child IDs from the list of NodeRelationship objects
            child_ids = [child.node_id for child in node.child_nodes]
            child_node_objects = docstore.get_nodes(child_ids)
            print(f"{indent}  - Children ({len(child_node_objects)}):")
            print_node_hierarchy(child_node_objects, docstore, level + 1)
        print() # Add a blank line for readability

# --- 1. Main Execution ---
if __name__ == "__main__":
    print("--- Inspecting Node Hierarchy ---")
    
    # Check if the database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Storage directory not found at '{DB_PATH}'.")
        print("Please run '1_build_database_advanced.py' first to create the database.")
        exit()

    # Load the storage context, which contains the docstore
    print(f"--- Loading docstore from persisted storage at '{DB_PATH}' ---")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=DB_PATH)
        docstore = storage_context.docstore
    except Exception as e:
        print(f"An error occurred while loading the storage context: {e}")
        exit()

    # Get all nodes from the docstore
    all_nodes_dict = docstore.docs
    all_nodes = list(all_nodes_dict.values())
    
    # Identify the root nodes (nodes that are not children of any other node)
    all_child_node_ids = set()
    for node in all_nodes:
        # FIX #2: Get child IDs from the list of NodeRelationship objects
        if node.child_nodes:
            child_ids = [child.node_id for child in node.child_nodes]
            all_child_node_ids.update(child_ids)
        
    root_nodes = [node for node in all_nodes if node.node_id not in all_child_node_ids]

    print(f"\nFound {len(all_nodes)} total nodes in the docstore.")
    print(f"Found {len(root_nodes)} root nodes to start the hierarchy.\n")
    print("-----------------------------------------")
    print("         DOCUMENT CHUNK HIERARCHY        ")
    print("-----------------------------------------\n")
    
    # Start printing the hierarchy from the root nodes
    print_node_hierarchy(root_nodes, docstore)