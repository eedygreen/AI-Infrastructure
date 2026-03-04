
import os
import openai 
import chromadb
from openai import OpenAI
from chromadb.config import Settings
from typing import Dict, List, Optional
from rich.console import Console
from pathlib import Path
from utils import logger
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

console = Console(stderr=True)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".").resolve()
    
    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and ((d / "chroma.sqlite3").exists() or "chroma" in d.name.lower())]
    chroma_dirs.sort(key=lambda d: d.name)
    
    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ) 
            )
            # Retrieve list of available collections from the database
            collections = client.list_collections()
                # Loop through each collection found
            for collection in collections:
                    # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}:{collection.name}"    
                    # Build information dictionary containing:
                info: Dict[str, str] = {
                    "path": str(chroma_dir),    # Store directory path as string
                    "collection": collection.name,  # Store collection name
                    "display_name": f"{chroma_dir.name}-{collection.name}", # Create user-friendly display name
                }     
                # Get document count with fallback for unsupported operations
                try:
                    count = collection.count()
                    info["count"]=str(count)
                    
                except Exception as e:
                    info["count"] = 0
                    logger.error(f"[rag_client] Error storing collection count: {type(e).__name__}")
                    
                # Add collection information to backends dictionary                
                backends[key]=info

        # Handle connection or access errors gracefully
        except Exception as e:
            err_msg = str(e)
            truncated_error = err_msg[:100]
            fallback_key = f"{chroma_dir.name}:error"
  
            backends[fallback_key] = {
                "path": str(chroma_dir),                # Create fallback entry for inaccessible directories
                "collection": "N/A",                    # Set appropriate fallback values for missing information
                "display_name": f"{chroma_dir.name} (error: {truncated_error})", # Include error information in display name with truncation
                "count": "0",                           # Set appropriate fallback values for missing information
                "error": err_msg,
            }
            logger.error(f"[rag_client]: {e}")

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chomadb persistentclient
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ) 
        )

        collection = client.get_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction(),
        )
        logger.info(f"✓ Successfully initialized collection: {collection_name}")
        return collection  # Return the collection with the collection_name
    except FileNotFoundError as e:
        logger.error(f"[rag_client] Error: Database directory not found at {chroma_dir}")
        return collection
    except Exception as e:
        logger.error(f"[rag_client] Error connecting to DB: {e}[")
        return None
    

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if openai_api_key.startswith("voc-"):
            client = OpenAI(
                api_key=openai_api_key,
                base_url="https://openai.vocareum.com/v1",
            )
        else:
            client = OpenAI(
                api_key=openai_api_key
            )
        logger.info(f"[rag_client] Successfully retrieved API Key")
    except Exception as e:
        logger.error(f"[rag_client] Error retrieving API Key: {e}")
        raise ValueError(f"[rag_client] Error with OPENAI client")

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )

        query_embedding = embedding_response.data[0].embedding  #List[float], dim=1536

        # Initialize filter variable to None (represents no filtering)
        where_filter = None
        # Check if filter parameter exists and is not set to "all" or equivalent
        
        if mission_filter and mission_filter.lower() != 'all':
        # If filter conditions are met, create filter dictionary with appropriate field-value pairs
            where_filter = {
                "mission": {"$eq": mission_filter}
            }
        # Execute database query with the following parameters:
        results = collection.query(
            query_embeddings=[query_embedding],     # Pass search query in the required format
            n_results=n_results,                    # Set maximum number of results to return
            where=where_filter                      # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        )

        if not results:
            logger.warning(f"[rag_client] Empty query returned")

        logger.info(f"[rag_client] Query return results")  
        return results                              # Return query results to caller
            
    except Exception as e:
        logger.error(f"[rag_client] Error retreiving documents: {e}", exc_info=True)
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts = ["=== RETREIVED CONTEXT ===\n"]

    # Loop through paired documents and their metadata using enumeration
    for id, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        # Extract mission information from metadata with fallback value
        mission = metadata.get('mission', "Unknown")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission.replace("_", " ").title()
        # Extract source information from metadata with fallback value 
        source = metadata.get("source", "Unknown source") 
        # Extract category information from metadata with fallback value
        category = metadata.get("category", "Uncategorized")
        # Clean up category name formatting (replace underscores, capitalize)
        category.replace("_", " ").title()
        # Create formatted source header with index number and extracted information
        source_header = f"[Documents {id}]\n    Missions:  {mission}  | Source: {source}  | Category:  {category}"
        # Add source header to context parts list
        context_parts.append(source_header)
        # Check document length and truncate if necessary
        max_doc_length = 500
        if len(doc) > max_doc_length:
            truncated_doc = f"{doc[:max_doc_length]} ...truncated_doc"
        # Add truncated or full document content to context parts list
            context_parts.append(truncated_doc)
        else:
            context_parts.append(doc)
        
        context_parts.append("\n")

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
"""
# testing
if __name__ == "__main__":
    backends = discover_chroma_backends()
    if backends:
        # Pick the first available backend
        first_key = list(backends.keys())[0]
        backend = backends[first_key]
        initialize_rag_system(chroma_dir=backend["path"], collection_name=backend["collection"])
    else:
        logger.error("No ChromaDB backends discovered.")

"""