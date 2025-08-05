"""
Knowledge Manager for Streamlit Inference Module
===============================================

Manages knowledge bases, document storage, and retrieval systems.
"""

import sqlite3
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """Manages knowledge bases and document storage."""
    
    def __init__(self, db_path: Path):
        """Initialize knowledge manager."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for knowledge management."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Knowledge bases table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    settings TEXT DEFAULT '{}'
                )
                ''')
                
                # Documents table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT DEFAULT 'text',
                    source_url TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
                )
                ''')
                
                # Document chunks table (for vector storage)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_vector TEXT,
                    chunk_metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
                ''')
                
                # Agent knowledge assignments
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_knowledge (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    assigned_by TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(agent_id, knowledge_base_id),
                    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
                )
                ''')
                
                # Search history
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    knowledge_base_id TEXT,
                    query TEXT NOT NULL,
                    results_count INTEGER DEFAULT 0,
                    search_metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
                )
                ''')
                
                conn.commit()
                logger.info("Knowledge database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing knowledge database: {e}")
            raise
    
    def create_knowledge_base(self, name: str, description: str, created_by: str, 
                            settings: Dict[str, Any] = None) -> Optional[str]:
        """Create a new knowledge base."""
        try:
            kb_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO knowledge_bases 
                (id, name, description, created_by, created_at, updated_at, settings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    kb_id, name, description, created_by, now, now,
                    json.dumps(settings or {})
                ))
                
                conn.commit()
                logger.info(f"Knowledge base '{name}' created with ID: {kb_id}")
                return kb_id
                
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return None
    
    def get_knowledge_bases(self, created_by: str = None) -> List[Dict[str, Any]]:
        """Get all knowledge bases, optionally filtered by creator."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if created_by:
                    cursor.execute('''
                    SELECT id, name, description, created_by, created_at, updated_at, settings
                    FROM knowledge_bases 
                    WHERE is_active = 1 AND created_by = ?
                    ORDER BY updated_at DESC
                    ''', (created_by,))
                else:
                    cursor.execute('''
                    SELECT id, name, description, created_by, created_at, updated_at, settings
                    FROM knowledge_bases 
                    WHERE is_active = 1
                    ORDER BY updated_at DESC
                    ''')
                
                kb_list = []
                for row in cursor.fetchall():
                    kb = {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "created_by": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                        "settings": json.loads(row[6] or '{}')
                    }
                    
                    # Get document count
                    cursor.execute('''
                    SELECT COUNT(*) FROM documents 
                    WHERE knowledge_base_id = ?
                    ''', (kb["id"],))
                    kb["document_count"] = cursor.fetchone()[0]
                    
                    kb_list.append(kb)
                
                return kb_list
                
        except Exception as e:
            logger.error(f"Error getting knowledge bases: {e}")
            return []
    
    def add_document(self, knowledge_base_id: str, title: str, content: str,
                    content_type: str = "text", source_url: str = None,
                    file_path: str = None, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add a document to a knowledge base."""
        try:
            doc_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if document with same hash already exists
                cursor.execute('''
                SELECT id FROM documents 
                WHERE knowledge_base_id = ? AND file_hash = ?
                ''', (knowledge_base_id, content_hash))
                
                if cursor.fetchone():
                    logger.warning(f"Document with same content already exists in knowledge base")
                    return None
                
                # Insert document
                cursor.execute('''
                INSERT INTO documents 
                (id, knowledge_base_id, title, content, content_type, source_url, 
                 file_path, file_hash, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id, knowledge_base_id, title, content, content_type,
                    source_url, file_path, content_hash,
                    json.dumps(metadata or {}), now, now
                ))
                
                # Create document chunks for better retrieval
                chunks = self._chunk_document(content)
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    cursor.execute('''
                    INSERT INTO document_chunks 
                    (id, document_id, chunk_index, content, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (chunk_id, doc_id, i, chunk, now))
                
                # Update knowledge base timestamp
                cursor.execute('''
                UPDATE knowledge_bases 
                SET updated_at = ? 
                WHERE id = ?
                ''', (now, knowledge_base_id))
                
                conn.commit()
                logger.info(f"Document '{title}' added to knowledge base {knowledge_base_id}")
                return doc_id
                
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return None
    
    def _chunk_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into chunks for better retrieval."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            # Find chunk end
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = -1
                
                for i in range(end, search_start, -1):
                    if content[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end != -1:
                    end = sentence_end
            
            # Extract chunk
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def search_knowledge(self, knowledge_base_id: str, query: str, 
                        limit: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """Search documents in a knowledge base."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Simple text search (in a real implementation, you'd use vector embeddings)
                search_query = f"%{query.lower()}%"
                
                cursor.execute('''
                SELECT d.id, d.title, d.content, d.content_type, d.source_url, 
                       d.metadata, dc.content as chunk_content, dc.chunk_index
                FROM documents d
                LEFT JOIN document_chunks dc ON d.id = dc.document_id
                WHERE d.knowledge_base_id = ? 
                AND (LOWER(d.content) LIKE ? OR LOWER(d.title) LIKE ? OR LOWER(dc.content) LIKE ?)
                ORDER BY 
                  CASE 
                    WHEN LOWER(d.title) LIKE ? THEN 1
                    WHEN LOWER(d.content) LIKE ? THEN 2
                    ELSE 3
                  END,
                  dc.chunk_index
                LIMIT ?
                ''', (knowledge_base_id, search_query, search_query, search_query, 
                      search_query, search_query, limit * 3))  # Get more chunks initially
                
                results = []
                seen_docs = set()
                
                for row in cursor.fetchall():
                    doc_id = row[0]
                    
                    if doc_id not in seen_docs and len(results) < limit:
                        result = {
                            "document_id": doc_id,
                            "title": row[1],
                            "content_preview": row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                            "content_type": row[3],
                            "source_url": row[4],
                            "metadata": json.loads(row[5] or '{}'),
                            "relevance_score": 0.5,  # Placeholder
                            "matched_chunk": row[6] if row[6] else row[2][:500]
                        }
                        results.append(result)
                        seen_docs.add(doc_id)
                
                # Record search
                if user_id:
                    search_id = str(uuid.uuid4())
                    cursor.execute('''
                    INSERT INTO search_history 
                    (id, user_id, knowledge_base_id, query, results_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (search_id, user_id, knowledge_base_id, query, len(results), 
                          datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"Knowledge search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    def assign_knowledge_to_agent(self, agent_id: str, knowledge_base_id: str, 
                                 assigned_by: str) -> bool:
        """Assign a knowledge base to an agent."""
        try:
            assignment_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT OR REPLACE INTO agent_knowledge 
                (id, agent_id, knowledge_base_id, assigned_by, assigned_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (assignment_id, agent_id, knowledge_base_id, assigned_by, now))
                
                conn.commit()
                logger.info(f"Knowledge base {knowledge_base_id} assigned to agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error assigning knowledge to agent: {e}")
            return False
    
    def get_agent_knowledge(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get knowledge bases assigned to an agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT kb.id, kb.name, kb.description, ak.assigned_at, ak.assigned_by
                FROM knowledge_bases kb
                JOIN agent_knowledge ak ON kb.id = ak.knowledge_base_id
                WHERE ak.agent_id = ? AND ak.is_active = 1 AND kb.is_active = 1
                ORDER BY ak.assigned_at DESC
                ''', (agent_id,))
                
                knowledge_list = []
                for row in cursor.fetchall():
                    knowledge_list.append({
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "assigned_at": row[3],
                        "assigned_by": row[4]
                    })
                
                return knowledge_list
                
        except Exception as e:
            logger.error(f"Error getting agent knowledge: {e}")
            return []
    
    def remove_agent_knowledge(self, agent_id: str, knowledge_base_id: str) -> bool:
        """Remove knowledge base assignment from agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE agent_knowledge 
                SET is_active = 0 
                WHERE agent_id = ? AND knowledge_base_id = ?
                ''', (agent_id, knowledge_base_id))
                
                conn.commit()
                logger.info(f"Knowledge base {knowledge_base_id} removed from agent {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error removing agent knowledge: {e}")
            return False
    
    def get_documents(self, knowledge_base_id: str) -> List[Dict[str, Any]]:
        """Get all documents in a knowledge base."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT id, title, content_type, source_url, created_at, updated_at, metadata
                FROM documents 
                WHERE knowledge_base_id = ?
                ORDER BY updated_at DESC
                ''', (knowledge_base_id,))
                
                documents = []
                for row in cursor.fetchall():
                    documents.append({
                        "id": row[0],
                        "title": row[1],
                        "content_type": row[2],
                        "source_url": row[3],
                        "created_at": row[4],
                        "updated_at": row[5],
                        "metadata": json.loads(row[6] or '{}')
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete chunks first
                cursor.execute('DELETE FROM document_chunks WHERE document_id = ?', (document_id,))
                
                # Delete document
                cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
                
                conn.commit()
                logger.info(f"Document {document_id} deleted")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        """Delete a knowledge base and all its content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all documents
                cursor.execute('SELECT id FROM documents WHERE knowledge_base_id = ?', 
                             (knowledge_base_id,))
                doc_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete chunks
                for doc_id in doc_ids:
                    cursor.execute('DELETE FROM document_chunks WHERE document_id = ?', (doc_id,))
                
                # Delete documents
                cursor.execute('DELETE FROM documents WHERE knowledge_base_id = ?', 
                             (knowledge_base_id,))
                
                # Delete agent assignments
                cursor.execute('DELETE FROM agent_knowledge WHERE knowledge_base_id = ?', 
                             (knowledge_base_id,))
                
                # Delete search history
                cursor.execute('DELETE FROM search_history WHERE knowledge_base_id = ?', 
                             (knowledge_base_id,))
                
                # Delete knowledge base
                cursor.execute('DELETE FROM knowledge_bases WHERE id = ?', (knowledge_base_id,))
                
                conn.commit()
                logger.info(f"Knowledge base {knowledge_base_id} deleted")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting knowledge base: {e}")
            return False
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge management statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total knowledge bases
                cursor.execute('SELECT COUNT(*) FROM knowledge_bases WHERE is_active = 1')
                total_kb = cursor.fetchone()[0]
                
                # Total documents
                cursor.execute('SELECT COUNT(*) FROM documents')
                total_docs = cursor.fetchone()[0]
                
                # Total chunks
                cursor.execute('SELECT COUNT(*) FROM document_chunks')
                total_chunks = cursor.fetchone()[0]
                
                # Recent searches
                cursor.execute('''
                SELECT COUNT(*) FROM search_history 
                WHERE datetime(created_at) > datetime('now', '-7 days')
                ''')
                recent_searches = cursor.fetchone()[0]
                
                return {
                    "total_knowledge_bases": total_kb,
                    "total_documents": total_docs,
                    "total_chunks": total_chunks,
                    "recent_searches": recent_searches
                }
                
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {
                "total_knowledge_bases": 0,
                "total_documents": 0,
                "total_chunks": 0,
                "recent_searches": 0
            }