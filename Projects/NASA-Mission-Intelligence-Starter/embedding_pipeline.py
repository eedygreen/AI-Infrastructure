#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os, re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils import logger

class ChunkingError(Exception):
    """Custom Eexception for chunking errors"""
    pass

class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""
    
    def __init__(self, 
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline
        
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer, go {chunk_size}"
            )
        if chunk_overlap < 0: 
            raise ValueError(
                f"chunk_overlap must be a positive integer, go {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less that the chunk_size ({chunk_size})"
            )
        
        try:

            # Initialize OpenAI client
            if openai_api_key.startswith("voc-"):
                self.client = OpenAI(
                    api_key=openai_api_key,
                    base_url="https://openai.vocareum.com/v1",
                )
            else:
                self.client = OpenAI(
                    api_key=openai_api_key
                )
            logger.info(f"Successfully retrieved API Key")
        except Exception as e:
            logger.error(f"Error retrieving API Key: {e}")

        # Store configuration parameters
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.open_api_key = openai_api_key

        try:
            # Initialize ChromaDB client
            self.chroma = chromadb.PersistentClient(
                path=self.chroma_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Successfully initialized ChromaDB client")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}", exc_info=True) 
            raise

        try:
            self.embedding_func = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=self.embedding_model
            )
            # Create or get collection
            self.collection = self.chroma.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func
            )

            logger.info(f"Collection '{self.collection_name}' ready!")

        except Exception as e:
            logger.error(f"Error creating collections '{self.collection_name}': {e}", exc_info=True)
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        try:
            if text is None:
                raise ValueError("text cannot be None")
            if not isinstance(text, str):
                raise TypeError(f"text must be a str, got {type(text).__name__}")
            
            if metadata is None:
                raise ValueError("metadata cannot be None")
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata must be dict, got {type(metadata).__name__}")
        
        except (TypeError, ValueError) as e:
            logger.error(f"Input Validation failed: {e}")
            raise
        
        if not text.strip():
            logger.warning("Empty text provided, returning empty list")
            return []
        
        try:
            text = text.replace('\00', '')
        except Exception as e:
            logger.warning(f"Text sanitization warning: {e}")
        result = []

        # Handle short texts that don't need chunking
        if len(text) <= self.chunk_size:
            try:
                enhanced_metadata = {
                    **metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_size": len(text)
                }
                result.append((text, enhanced_metadata))
                return result
            except Exception as e:
                logger.error(f"Error creating single chunk: {e}")
                raise ChunkingError(f"Failed to create Single Chunk: {e}")
        
        # chunking logic with overlap
        try: 
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""] # break at sentence boundaries
                )
            except Exception as e:
                logger.error(f"Failed to initialize splitter: {e}")
                raise ChunkingError(f"Failed to create splitter: {e}")
            
            try:
                documents = splitter.create_documents(
                    texts=[text],
                    metadatas=[metadata]
                ) 
            except Exception as e:
                logger.error(f"create_documents failed: {e}")
                raise ChunkingError(f"Failed to create documents: {e}")

            if not documents:
                raise ChunkingError("No documents were created")
            
            for i, doc in enumerate(documents):
                try:
                    chunk_text = doc.page_content
                    if not chunk_text or not chunk_text.strip(): 
                        logger.warning(f"Skipping empty chunk at index {i}")
                        continue

                    chunk_metadata = {
                        **doc.metadata,         # Create metadata for each chunk
                        "chunk_index": i + 1,
                        "total_chunks": len(documents),
                        "chunk_size": len(chunk_text)
                    }

                    result.append((chunk_text, chunk_metadata))

                except AttributeError as e:
                    logger.error(f"Document {i} missing expected atrributes: {e}")
                    continue

                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    continue

            if not result:
                raise ChunkingError("No valid chunks were created")

            return result
        
        except ImportError as e:
            logger.error("LangChain import failed")
            raise ChunkingError(
                "LangChain is required but not installed. \n"
                "Install with: pip install langchain"
            ) 
        
        except ChunkingError:
            raise

        except Exception as e:
            logger.error(f"Unexpected error during chunking: {e}", exc_info=True)
            raise ChunkingError(f"Chunking failed with unexpected error: {e}") 
    
    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            if doc_id is None:
                raise ValueError("doc_id cannot be None")
            if not isinstance(doc_id, str):
                raise TypeError(f"doc_id must be str, got {type(doc_id).__name__}")
            if not doc_id.strip():
                raise ValueError("doc_id cannot be empty or whitespace")
            
            doc_id = doc_id.strip()

        except (ValueError, TypeError) as e:
            logger.error(f"doc_id Validation failed: {e}", exc_info=True)
            raise
        
        try:
            logger.debug(f"Checking if document exist")
            # Query collection for document ID
            result = self.collection.get(ids=[doc_id], include=[])
            # Return True if exists, False otherwise
            exists = len(result.get('ids'), []) > 0

            logger.info(f"Document {'exists' if exists else 'not found'}: {doc_id}")

            return exists
        except Exception as e:
            logger.error(f"Query failed for {doc_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to check '{doc_id}'") 
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection
        
        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)
            
            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}", exc_info=True)
            return False
    
    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source (useful for re-processing files)
        
        Args:
            source_pattern: Pattern to match source names
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching source pattern: {source_pattern}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}", exc_info=True)
            return 0
    
    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('source') == source and 
                    metadata.get('mission') == mission):
                    file_doc_ids.append(all_docs['ids'][i])
            
            return file_doc_ids
            
        except Exception as e:
            logger.error(f"Error getting file documents: {e}", exc_info=True)
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """

        if not text or not isinstance(text, str):
            raise ValueError(f"Invalide text: {text}")
        
        text = text.strip()
        # Call OpenAI embeddings API
        try:
            logger.debug(f"Generating embedding lenght: {len(text)}")

            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model,
            )
            embedding = response.data[0].embedding

            if not embedding:
                raise RuntimeError("Empty embedding returned")
            
            logger.info(f"Embeddings successfully generated! Dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise 
    
    def _sanitised(self, value: str) -> str:
        """
        Remove special chars, replaced whitespace with '_', and ensure lowercase string is returned.
        
        Args:
            value: The string(s) to sanitised
        
        Return
            str: The sanitised string
        """
        sanitized = str(value).lower().replace(' ', '_').replace('-', '_')
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs

        Args:
            file_path
            metadata: 
        
        Returns:
            str: doc_id with format (mission_source_chunk_0001)
        """
        if file_path is None:
            raise ValueError("file_path cannot be empty string")
        if not isinstance(file_path, (Path, str)):
            raise TypeError(f"file_path {file_path} must be a Path or string")
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if metadata is None:
            raise ValueError("metadata cannot be Null")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        mission = metadata.get('mission')
        source = metadata.get('source', file_path.stem)
        chunk_index = metadata.get('chunk_index')

        if not mission:
            logger.error(f"missing mission: {list(metadata.keys())}", exc_info=True)
            raise ValueError(f"Missing mission")
        
        if chunk_index is None:
            logger.error(f"Null chunk index: {list(metadata.keys())}", exc_info=True)
            raise ValueError(f"Missing chunk index")
        
        chunk_index = int(chunk_index)
        if chunk_index < 0:
            raise ValueError(f"chunk_index must be be non-negative, got {chunk_index}")

        chunk_str = f"{chunk_index:.04d}"
        mission_sanitized = self._sanitised(mission)
        source_sanitized = self._sanitised(source)

        doc_id = f"{mission_sanitized}_{source_sanitized}_chunk_{chunk_str}"
        
        logger.info(f"Generated ID: {doc_id}")
        return doc_id
    
    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            # Enhanced metadata extraction
            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return self.chunk_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
    
    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'
    
    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'
    
    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()
        
        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'
        
        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'
        
        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'
        
        # General categories
        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'
    
    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)
        
        Args:
            base_path: Base directory path
            
        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []
        
        # Define directories to scan
        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]
        
        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                
                # Find only text files
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")
        
        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (file_path.name.startswith('.') or 
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'):
                continue
            filtered_files.append(file_path)
        
        logger.info(f"Total text files to process: {len(filtered_files)}")
        
        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1
        
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")
        
        return filtered_files
    
    def add_documents_to_collection(self, documents: List[Tuple[str, Dict[str, Any]]], 
                                   file_path: Path, batch_size: int = 50, 
                                   update_mode: str = 'skip') -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling
        
        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if documents is None:
            raise ValueError("documents cannot be None")
        
        if not isinstance(documents, list):
            raise TypeError(f"documents must be a list, got {type(documents).__name__}")
        
        if not documents:
            logger.info("No documents to add - empty list received.")
            return {'added': 0, 'updated': 0, 'skipped': 0}
        
        if file_path is None:
            raise ValueError("file_path cannot be None")
        
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive int, got {batch_size}")
        
        valid_modes = ['skip', 'update', 'replace']
        if update_mode not in valid_modes:
            raise ValueError(f"update_mode must be one of these {valid_modes}, got '{update_mode}")
        
        # Handle different update modes (skip, update, replace)

        if update_mode == 'replace':
            try:
                logger.info(f"Replace mode: deleting existing docs from {file_path.name}")
                deleted = self.delete_documents_by_source(file_path)
                logger.info(
                    f"[replace] Deleted {deleted} existing documents")

            except Exception as e:
                logger.error(f"Failed to delete existing documents from {file_path}: {e}", exc_info=True)
                raise RuntimeError("Deletion failed") 
        
        stats: Dict[str, int] = {'added': 0, 'updated': 0, 'skipped': 0}
        start_time = time.time()
        total_docs = len(documents)

        logger.info(
            f"Starting ingestion: {total_docs} docs, | "
            f"batch_size={batch_size} | mode={update_mode} file={file_path.name}"
        )
        
        ids_to_add: List[str] = []
        texts_to_add: List[str] = []
        metas_to_add: List[Dict[str, Any]] = []

        ids_to_update: List[str] = []
        texts_to_update: List[str] = []
        metas_to_update: List[Dict[str, Any]] = []
        
        for text, metadata in documents:
            doc_id = self.generate_document_id(file_path, metadata)     # Generate document ID
            exists: bool = self.check_document_exists(doc_id)           # Check if exists

            if update_mode == 'skip':
                if exists:
                    stats['skipped'] += 1
                    logger.debug(f"['skip'] Document '{doc_id}' already exists - skipping.")         
                else:
                    ids_to_add.append(doc_id)
                    texts_to_add.append(text)
                    metas_to_add.append(metadata)

            elif update_mode == 'update':
                if exists:
                    ids_to_update.append(doc_id)
                    texts_to_update.append(text)
                    metas_to_update.append(metadata)
                else:
                    ids_to_add.append(doc_id)
                    texts_to_add.append(text)
                    metas_to_add.append(metadata)

            else:   # This replace all pre-existing docs were deleted above
                ids_to_add.append(doc_id)
                texts_to_add.append(text)
                metas_to_add.append(metadata)

        def _batch_op(
            ids: List[str],
            texts: List[str],
            metas: List[Dict[str, Any]],
            operation: str
        ) -> int:
            """ Execute collection.add or collection.update in batch_size slices.
                Embeddins are generated in bulk per batch, not one-by-one,
                Add documents in batches; returns count of successfully.

                Args:
                    ids: List of strings
                    texts: List strings
                    metas: Metadatas, List of key-value pairs
                    operation: string [Add, Update]
                
                Return:
                    stats - track of operations [Add, Update] as the processed batches
            """
            collection_op = (self.collection.add if operation == 'add' else self.collection.update)
            count = 0
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                batch_ids = ids[start: end]
                batch_texts = texts[start: end]
                batch_metas = metas[start: end]

                try:
                    embeddings = [self.get_embedding(t) for t in batch_texts]
                    
                    # collection: add or update
                    collection_op(
                        ids = batch_ids,
                        documents = batch_texts,
                        embeddings = embeddings,
                        metadatas = batch_metas,
                    )
                    count += len(batch_ids)
                    logger.info(
                        f"[{operation}] Batch {start // batch_size + 1}: "
                        f"{len(batch_ids)} document(s)."
                    )
                except Exception as e:
                    logger.error(
                        f"[{operation}] Batch at index {start} failed: {e}", exc_info=True
                    )
                    raise RuntimeError(f"Batch {operation} failed at index {start}") 
                
            return count
        
        if ids_to_add:
            stats['added'] = _batch_op(ids_to_add, texts_to_add, metas_to_add, 'add')

        if ids_to_update:
            stats["updated"] = _batch_op(ids_to_update, texts_to_update, metas_to_update, 'update')

        elapsed = time.time() - start_time
        logger.info(
            f"Ingestion complete in {elapsed:.2f}s - "
            f"added={stats['added']}, updated={stats['updated']}, skipped={stats['skipped']}"
        )
            # Return statistics
        return stats
        
    def process_all_text_data(self, base_path: str, update_mode: str = 'skip') -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB
        
        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents from file and re-add
            
        Returns:
            Statistics about processed files
        """
        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        # Get files to process
        try:
            files_to_process: List[Path] = self.scan_text_files_only(base_path)
        except Exception as e:
            logger.error(f"Failed to scan '{base_path} for text files.", exc_info=True)
            raise RuntimeError(f"Directory scan failed for '{base_path}") 
        
        if not files_to_process:
            logger.warning(f"No text files found under '{base_path}'.")
            return stats
        
        logger.info(
            f"Found {len(files_to_process)} file(s) to process '{base_path}'."
        )

        # Loop through each file
        for file_path in files_to_process:
            file_name = Path(file_path).name
            logger.info(f"Processing '{file_name}'...")

            try:
                # Process file and add to collection
                documents: List[Tuple[str, Dict[str, Any]]] = (
                    self.process_text_file(file_path)
                )

                if not documents:
                    logger.warning(f"'{file_name}' produced no chunks - skipping...")
                    continue
                
                file_stats = self.add_documents_to_collection(
                    documents=documents,
                    file_path=Path(file_path),
                    update_mode=update_mode
                )
                # Update statistics
                stats['files_processed'] += 1
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']
                stats['total_chunks'] += len(documents)
                stats['missions'][file_name] ={
                    'chunks': len(documents),
                    'added': file_stats['added'],
                    'updated': file_stats['updated'],
                    'skipped': file_stats['skipped']
                } 

                logger.info(
                    f"'{file_name}' - chunks={len(documents)}, "
                    f"added={file_stats['added']}, "
                    f"updated={file_stats['updated']}, "
                    f"skipped={file_stats['skipped']}"
                )
               # Handle errors gracefully
            except Exception as e:
                stats['files_errored'] += 1
                stats['missions'][file_name] = {'error': str(e)}
                logger.error(f"Failed to process '{file_name}': {e}", exc_info=True)
                continue
        
        logger.info(
            f"Done. files_processed={stats['files_processed']}, "
            f"files_errored={stats['files_errored']}, "
            f"total_chunks={stats['total_chunks']}, "
            f"added={stats['documents_added']}, "
            f"updated={stats['documents_updated']}, "
            f"skipped={stats['documents_skipped']}"
        )
    
        return stats
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection"""
        # Return collection name, document count, metadata
        if not hasattr(self.collection, 'name'):
            raise RuntimeError(
                f"Collection is not initalised - got {type(self.collection).__name__}."
                f"ensure setup completed before querying."
            )
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata,
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for testing
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            Query results
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty.")
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValueError(f"n_results must be a positive int, go {n_results}")
        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Collection is empty. Cannot perform query")
                return 

            n_results = min(n_results, collection_count)

            embeddings: List[float] = self.get_embedding(query_text)
            # Perform test query and return results
            result = self.collection.query(
                query_embeddings = [embeddings],
                n_results = n_results,
                include = ["documents", "metadatas", "distances"],
            )

            if not result:
                raise RuntimeError(f"Query returned empty")
            
            logger.debug(f"Query returned - {len(result.keys())} items")
            return result
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise   
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection"""
        try:
            # Get all documents to analyze
            all_docs = self.collection.get()
            
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}
            
            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')
                
                # Count by mission
                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                
                # Count by data type
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                
                # Count by document category
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                
                # Count by file type
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', default=os.getenv('OPENAI_API_KEY'), help='OpenAI API key (can also be set via OPENAI_API_KEY env var)')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                       help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    if not args.openai_key:
        parser.error("the following arguments are required: --openai-key (or set OPENAI_API_KEY environment variable)")

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}")
        return
    
    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return
    
    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    
    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}")
    
    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")
    
    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query, 3)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):  # Show top 3
                logger.info(f"Result {i+1}: {doc[:200]}...")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
  