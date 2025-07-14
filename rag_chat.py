"""
RAG-Enhanced Chat System
Simple RAG integration for chat interface with document upload and context-aware responses
"""

import os
import logging
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Document processing
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

# Vector database
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class SimpleRAGChat:
    """Simple RAG-enhanced chat system"""
    
    def __init__(self, persist_directory="./rag_data"):
        self.persist_directory = persist_directory
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_chunks_per_query = 3
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self.documents = {}  # Simple in-memory document storage
        
        self._initialize_chroma()
        logger.info("Simple RAG Chat initialized")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name="chat_documents")
                logger.info("Loaded existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(name="chat_documents")
                logger.info("Created new ChromaDB collection")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def add_document_content(self, content: str, filename: str, file_type: str = "text") -> str:
        """Add document content to the RAG system"""
        try:
            if not self.collection:
                return None
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Store document info
            self.documents[doc_id] = {
                'id': doc_id,
                'filename': filename,
                'file_type': file_type,
                'content': content,
                'added_at': datetime.now().isoformat(),
                'chunk_count': 0
            }
            
            # Split content into chunks
            chunks = self._split_text(content)
            
            # Add chunks to ChromaDB
            chunk_ids = []
            chunk_contents = []
            chunk_metadata = []
            
            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk_ids.append(chunk_id)
                chunk_contents.append(chunk_content)
                chunk_metadata.append({
                    'document_id': doc_id,
                    'document_name': filename,
                    'file_type': file_type,
                    'chunk_index': i
                })
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunk_contents,
                ids=chunk_ids,
                metadatas=chunk_metadata
            )
            
            # Update document info
            self.documents[doc_id]['chunk_count'] = len(chunks)
            
            logger.info(f"Added document {filename} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document content: {e}")
            return None
    
    def process_uploaded_file(self, file_path: str, filename: str) -> Optional[str]:
        """Process an uploaded file and add to RAG system"""
        try:
            # Determine file type
            file_extension = os.path.splitext(filename)[1].lower()
            file_type = self._get_file_type(file_extension)
            
            # Extract text content
            text_content = self._extract_text(file_path, file_type)
            
            if not text_content:
                logger.warning(f"No text content extracted from {filename}")
                return None
            
            # Add to RAG system
            return self.add_document_content(text_content, filename, file_type)
            
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")
            return None
    
    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension"""
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.txt': 'text',
            '.md': 'markdown',
            '.html': 'html',
            '.htm': 'html',
            '.json': 'json',
            '.csv': 'csv'
        }
        return type_mapping.get(extension, 'text')
    
    def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text content from various file types"""
        try:
            if file_type == 'pdf':
                return self._extract_pdf_text(file_path)
            elif file_type == 'docx':
                return self._extract_docx_text(file_path)
            elif file_type == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_type == 'markdown':
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                    html = markdown.markdown(md_content)
                    return BeautifulSoup(html, 'html.parser').get_text()
            elif file_type == 'html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                    return BeautifulSoup(html, 'html.parser').get_text()
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            elif file_type == 'csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return ""
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return ""
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {e}")
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            if not self.collection:
                logger.error("ChromaDB collection not initialized")
                return []
            
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.max_chunks_per_query)
            )
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(search_results)} relevant chunks for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def enhance_query_with_context(self, query: str, system_message: str = None) -> tuple[str, List[Dict[str, Any]]]:
        """Enhance query with relevant document context"""
        try:
            # Search for relevant documents
            search_results = self.search_documents(query)
            
            if not search_results:
                return query, []
            
            # Generate context from search results
            context_parts = []
            for result in search_results:
                content = result['content']
                metadata = result['metadata']
                
                context_part = f"""
Document: {metadata.get('document_name', 'Unknown')}
Relevance: {result['similarity_score']:.2f}
Content: {content[:500]}...
---
"""
                context_parts.append(context_part)
            
            # Create enhanced query with context
            enhanced_query = f"""
Based on the following relevant documents, please provide a comprehensive answer to the user's question.

User Query: {query}

Relevant Context:
{''.join(context_parts)}

Instructions: Please provide a detailed response based on the above context. If the context doesn't contain enough information to answer the question completely, please mention that and provide what information is available from the context.
"""
            
            return enhanced_query, search_results
            
        except Exception as e:
            logger.error(f"Failed to enhance query with context: {e}")
            return query, []
    
    def get_documents_list(self) -> List[Dict[str, Any]]:
        """Get list of uploaded documents"""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            if doc_id not in self.documents:
                return False
            
            # Delete from ChromaDB
            if self.collection:
                # Find all chunks for this document
                chunk_count = self.documents[doc_id]['chunk_count']
                chunk_ids = [f"{doc_id}_{i}" for i in range(chunk_count)]
                
                try:
                    self.collection.delete(ids=chunk_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete chunks from ChromaDB: {e}")
            
            # Delete from documents dict
            del self.documents[doc_id]
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            total_docs = len(self.documents)
            total_chunks = sum(doc['chunk_count'] for doc in self.documents.values())
            
            file_types = {}
            for doc in self.documents.values():
                file_type = doc['file_type']
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'file_types': file_types,
                'collection_initialized': self.collection is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

# Global RAG chat instance
rag_chat = None

def get_rag_chat() -> SimpleRAGChat:
    """Get or create global RAG chat instance"""
    global rag_chat
    if rag_chat is None:
        rag_chat = SimpleRAGChat()
    return rag_chat