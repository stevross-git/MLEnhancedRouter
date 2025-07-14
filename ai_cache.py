"""
AI Response Caching System
Provides efficient caching for AI model responses to reduce API calls and improve performance
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sqlite3
import logging
from sqlalchemy import and_, func
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached AI response"""
    key: str
    query: str
    response: str
    model_id: str
    system_message: Optional[str]
    created_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any]
    hit_count: int = 0
    last_accessed: Optional[datetime] = None

class AICacheManager:
    """Manages AI response caching with database backend"""
    
    def __init__(self, db=None, ttl_seconds: int = 3600, max_size: int = 10000):
        """
        Initialize cache manager
        
        Args:
            db: Database instance (Flask-SQLAlchemy)
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of entries in cache
        """
        self.db = db
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Import models here to avoid circular imports
        from models import AICacheEntry, AICacheStats
        self.AICacheEntry = AICacheEntry
        self.AICacheStats = AICacheStats
        
        logger.info(f"AI Cache initialized with database backend, TTL: {ttl_seconds}s, max size: {max_size}")
    
    def _init_sqlite(self):
        """Initialize SQLite cache storage"""
        self.db_path = os.path.join(os.getcwd(), "ai_cache.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ai_cache (
                key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                model_id TEXT NOT NULL,
                system_message TEXT,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                metadata TEXT,
                hit_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_expires_at ON ai_cache(expires_at)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_id ON ai_cache(model_id)
        ''')
        self.conn.commit()
    
    def _init_redis(self):
        """Initialize Redis cache storage"""
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis cache backend initialized")
        except ImportError:
            logger.warning("Redis not available, falling back to SQLite")
            self.cache_type = "sqlite"
            self._init_sqlite()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to SQLite")
            self.cache_type = "sqlite"
            self._init_sqlite()
    
    def _generate_cache_key(self, query: str, model_id: str, system_message: Optional[str] = None) -> str:
        """Generate a unique cache key for the query"""
        content = f"{query}|{model_id}|{system_message or ''}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, query: str, model_id: str, system_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response
        
        Args:
            query: The query string
            model_id: AI model identifier
            system_message: Optional system message
            
        Returns:
            Cached response data or None if not found/expired
        """
        if not self.db:
            return None
            
        cache_key = self._generate_cache_key(query, model_id, system_message)
        
        try:
            # Query the database for the cache entry
            entry = self.db.session.query(self.AICacheEntry).filter(
                and_(
                    self.AICacheEntry.cache_key == cache_key,
                    self.AICacheEntry.expires_at > datetime.now()
                )
            ).first()
            
            if entry:
                # Update hit count and last accessed
                entry.increment_hit_count()
                self.db.session.commit()
                
                return {
                    'response': entry.response,
                    'model_id': entry.model_id,
                    'cached_at': entry.created_at.isoformat(),
                    'metadata': entry.meta_data or {},
                    'hit_count': entry.hit_count
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from database cache: {e}")
            return None
    
    def set(self, query: str, model_id: str, response: str, system_message: Optional[str] = None, 
            metadata: Dict[str, Any] = None) -> bool:
        """
        Store response in cache
        
        Args:
            query: The query string
            model_id: AI model identifier
            response: AI response to cache
            system_message: Optional system message
            metadata: Additional metadata
            
        Returns:
            True if cached successfully
        """
        if not self.db:
            return False
            
        cache_key = self._generate_cache_key(query, model_id, system_message)
        
        try:
            # Check if entry already exists
            existing = self.db.session.query(self.AICacheEntry).filter(
                self.AICacheEntry.cache_key == cache_key
            ).first()
            
            if existing:
                # Update existing entry
                existing.response = response
                existing.expires_at = datetime.now() + timedelta(seconds=self.ttl_seconds)
                existing.meta_data = metadata or {}
            else:
                # Create new entry
                entry = self.AICacheEntry(
                    cache_key=cache_key,
                    query=query,
                    response=response,
                    model_id=model_id,
                    system_message=system_message,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds),
                    meta_data=metadata or {}
                )
                self.db.session.add(entry)
            
            self.db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing in database cache: {e}")
            self.db.session.rollback()
            return False
    
    def _cleanup_expired(self):
        """Clean up expired entries from database"""
        if not self.db:
            return
            
        try:
            expired_count = self.db.session.query(self.AICacheEntry).filter(
                self.AICacheEntry.expires_at < datetime.now()
            ).delete()
            
            self.db.session.commit()
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            self.db.session.rollback()
    
    def clear(self, model_id: Optional[str] = None):
        """Clear cache entries"""
        if not self.db:
            return
            
        try:
            if model_id:
                self.db.session.query(self.AICacheEntry).filter(
                    self.AICacheEntry.model_id == model_id
                ).delete()
            else:
                self.db.session.query(self.AICacheEntry).delete()
            
            self.db.session.commit()
            logger.info(f"Cleared cache entries for model: {model_id or 'all'}")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            self.db.session.rollback()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_type': 'database',
            'ttl_seconds': self.ttl_seconds,
            'max_size': self.max_size
        }
        
        if not self.db:
            return stats
            
        try:
            # Get total entries
            total_entries = self.db.session.query(self.AICacheEntry).count()
            
            # Get valid (non-expired) entries
            valid_entries = self.db.session.query(self.AICacheEntry).filter(
                self.AICacheEntry.expires_at > datetime.now()
            ).count()
            
            # Get average hit count
            avg_hit_count = self.db.session.query(func.avg(self.AICacheEntry.hit_count)).scalar() or 0
            
            # Get total hits
            total_hits = self.db.session.query(func.sum(self.AICacheEntry.hit_count)).scalar() or 0
            
            stats.update({
                'total_entries': total_entries,
                'valid_entries': valid_entries,
                'expired_entries': total_entries - valid_entries,
                'average_hit_count': round(float(avg_hit_count), 2),
                'total_hits': total_hits,
                'hit_rate': round((total_hits / max(total_entries, 1)) * 100, 2)
            })
            
        except Exception as e:
            logger.error(f"Error getting database cache stats: {e}")
            
        return stats
    
    def get_cache_entries(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent cache entries for debugging/monitoring"""
        entries = []
        
        if not self.db:
            return entries
            
        try:
            query = self.db.session.query(self.AICacheEntry).filter(
                self.AICacheEntry.expires_at > datetime.now()
            )
            
            if model_id:
                query = query.filter(self.AICacheEntry.model_id == model_id)
            
            query = query.order_by(self.AICacheEntry.created_at.desc()).limit(limit)
            
            for entry in query.all():
                query_preview = entry.query[:100] + '...' if len(entry.query) > 100 else entry.query
                entries.append({
                    'key': entry.cache_key,
                    'query': query_preview,
                    'model_id': entry.model_id,
                    'created_at': entry.created_at.isoformat(),
                    'expires_at': entry.expires_at.isoformat(),
                    'hit_count': entry.hit_count,
                    'last_accessed': entry.last_accessed.isoformat() if entry.last_accessed else None
                })
                
        except Exception as e:
            logger.error(f"Error getting cache entries: {e}")
            
        return entries

# Global cache instance
cache_manager = None

def get_cache_manager(db=None) -> AICacheManager:
    """Get or create global cache manager instance"""
    global cache_manager
    if cache_manager is None:
        ttl_seconds = int(os.getenv("AI_CACHE_TTL", "3600"))
        max_size = int(os.getenv("AI_CACHE_MAX_SIZE", "10000"))
        
        cache_manager = AICacheManager(
            db=db,
            ttl_seconds=ttl_seconds,
            max_size=max_size
        )
    elif db is not None and cache_manager.db is None:
        # Update existing cache manager with database instance
        cache_manager.db = db
        # Re-import models with proper database context
        from models import AICacheEntry, AICacheStats
        cache_manager.AICacheEntry = AICacheEntry
        cache_manager.AICacheStats = AICacheStats
    
    return cache_manager