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
    """Manages AI response caching with multiple storage backends"""
    
    def __init__(self, cache_type: str = "sqlite", ttl_seconds: int = 3600, max_size: int = 10000):
        """
        Initialize cache manager
        
        Args:
            cache_type: Type of cache storage ('sqlite', 'memory', 'redis')
            ttl_seconds: Time to live for cache entries in seconds
            max_size: Maximum number of entries in cache
        """
        self.cache_type = cache_type
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Initialize storage backend
        if cache_type == "sqlite":
            self._init_sqlite()
        elif cache_type == "redis":
            self._init_redis()
        
        logger.info(f"AI Cache initialized with {cache_type} backend, TTL: {ttl_seconds}s, max size: {max_size}")
    
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
        cache_key = self._generate_cache_key(query, model_id, system_message)
        
        if self.cache_type == "memory":
            return self._get_from_memory(cache_key)
        elif self.cache_type == "sqlite":
            return self._get_from_sqlite(cache_key)
        elif self.cache_type == "redis":
            return self._get_from_redis(cache_key)
        
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
        cache_key = self._generate_cache_key(query, model_id, system_message)
        
        entry = CacheEntry(
            key=cache_key,
            query=query,
            response=response,
            model_id=model_id,
            system_message=system_message,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.ttl_seconds),
            metadata=metadata or {}
        )
        
        if self.cache_type == "memory":
            return self._set_in_memory(cache_key, entry)
        elif self.cache_type == "sqlite":
            return self._set_in_sqlite(cache_key, entry)
        elif self.cache_type == "redis":
            return self._set_in_redis(cache_key, entry)
        
        return False
    
    def _get_from_memory(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get entry from memory cache"""
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if datetime.now() < entry.expires_at:
                entry.hit_count += 1
                entry.last_accessed = datetime.now()
                return {
                    'response': entry.response,
                    'model_id': entry.model_id,
                    'cached_at': entry.created_at.isoformat(),
                    'metadata': entry.metadata
                }
            else:
                del self.memory_cache[cache_key]
        return None
    
    def _set_in_memory(self, cache_key: str, entry: CacheEntry) -> bool:
        """Set entry in memory cache"""
        # Check size limit
        if len(self.memory_cache) >= self.max_size:
            # Remove oldest entries
            oldest_keys = sorted(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k].created_at)[:100]
            for key in oldest_keys:
                del self.memory_cache[key]
        
        self.memory_cache[cache_key] = entry
        return True
    
    def _get_from_sqlite(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get entry from SQLite cache"""
        try:
            cursor = self.conn.execute('''
                SELECT response, model_id, created_at, metadata, hit_count
                FROM ai_cache 
                WHERE key = ? AND expires_at > ?
            ''', (cache_key, datetime.now()))
            
            row = cursor.fetchone()
            if row:
                # Update hit count and last accessed
                self.conn.execute('''
                    UPDATE ai_cache 
                    SET hit_count = hit_count + 1, last_accessed = ?
                    WHERE key = ?
                ''', (datetime.now(), cache_key))
                self.conn.commit()
                
                return {
                    'response': row[0],
                    'model_id': row[1],
                    'cached_at': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {},
                    'hit_count': row[4] + 1
                }
        except Exception as e:
            logger.error(f"Error getting from SQLite cache: {e}")
        
        return None
    
    def _set_in_sqlite(self, cache_key: str, entry: CacheEntry) -> bool:
        """Set entry in SQLite cache"""
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO ai_cache 
                (key, query, response, model_id, system_message, created_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                entry.query,
                entry.response,
                entry.model_id,
                entry.system_message,
                entry.created_at,
                entry.expires_at,
                json.dumps(entry.metadata)
            ))
            self.conn.commit()
            
            # Clean up expired entries periodically
            if hash(cache_key) % 100 == 0:  # Clean up every 100th insert
                self._cleanup_expired()
            
            return True
        except Exception as e:
            logger.error(f"Error setting in SQLite cache: {e}")
            return False
    
    def _get_from_redis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get entry from Redis cache"""
        try:
            data = self.redis_client.get(f"ai_cache:{cache_key}")
            if data:
                entry = pickle.loads(data)
                if datetime.now() < entry.expires_at:
                    # Update hit count
                    entry.hit_count += 1
                    entry.last_accessed = datetime.now()
                    self.redis_client.setex(
                        f"ai_cache:{cache_key}",
                        self.ttl_seconds,
                        pickle.dumps(entry)
                    )
                    
                    return {
                        'response': entry.response,
                        'model_id': entry.model_id,
                        'cached_at': entry.created_at.isoformat(),
                        'metadata': entry.metadata,
                        'hit_count': entry.hit_count
                    }
                else:
                    self.redis_client.delete(f"ai_cache:{cache_key}")
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
        
        return None
    
    def _set_in_redis(self, cache_key: str, entry: CacheEntry) -> bool:
        """Set entry in Redis cache"""
        try:
            self.redis_client.setex(
                f"ai_cache:{cache_key}",
                self.ttl_seconds,
                pickle.dumps(entry)
            )
            return True
        except Exception as e:
            logger.error(f"Error setting in Redis cache: {e}")
            return False
    
    def _cleanup_expired(self):
        """Clean up expired entries from SQLite"""
        try:
            cursor = self.conn.execute('''
                DELETE FROM ai_cache WHERE expires_at < ?
            ''', (datetime.now(),))
            deleted_count = cursor.rowcount
            self.conn.commit()
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
    
    def clear(self, model_id: Optional[str] = None):
        """Clear cache entries"""
        if self.cache_type == "memory":
            if model_id:
                keys_to_remove = [k for k, v in self.memory_cache.items() if v.model_id == model_id]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()
        
        elif self.cache_type == "sqlite":
            try:
                if model_id:
                    self.conn.execute('DELETE FROM ai_cache WHERE model_id = ?', (model_id,))
                else:
                    self.conn.execute('DELETE FROM ai_cache')
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error clearing SQLite cache: {e}")
        
        elif self.cache_type == "redis":
            try:
                if model_id:
                    # This would require scanning all keys, which is expensive
                    # In practice, you'd want to use a more efficient approach
                    pass
                else:
                    for key in self.redis_client.scan_iter(match="ai_cache:*"):
                        self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_type': self.cache_type,
            'ttl_seconds': self.ttl_seconds,
            'max_size': self.max_size
        }
        
        if self.cache_type == "memory":
            stats.update({
                'total_entries': len(self.memory_cache),
                'memory_usage_mb': sum(len(str(v).encode()) for v in self.memory_cache.values()) / (1024 * 1024)
            })
        
        elif self.cache_type == "sqlite":
            try:
                cursor = self.conn.execute('SELECT COUNT(*) FROM ai_cache')
                total_entries = cursor.fetchone()[0]
                
                cursor = self.conn.execute('SELECT COUNT(*) FROM ai_cache WHERE expires_at > ?', (datetime.now(),))
                valid_entries = cursor.fetchone()[0]
                
                cursor = self.conn.execute('SELECT AVG(hit_count) FROM ai_cache WHERE hit_count > 0')
                avg_hit_count = cursor.fetchone()[0] or 0
                
                stats.update({
                    'total_entries': total_entries,
                    'valid_entries': valid_entries,
                    'expired_entries': total_entries - valid_entries,
                    'average_hit_count': round(avg_hit_count, 2)
                })
            except Exception as e:
                logger.error(f"Error getting SQLite stats: {e}")
        
        elif self.cache_type == "redis":
            try:
                keys = list(self.redis_client.scan_iter(match="ai_cache:*"))
                stats.update({
                    'total_entries': len(keys),
                    'redis_memory_usage': self.redis_client.memory_usage('ai_cache:*') if keys else 0
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        
        return stats
    
    def get_cache_entries(self, model_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent cache entries for debugging/monitoring"""
        entries = []
        
        if self.cache_type == "sqlite":
            try:
                if model_id:
                    cursor = self.conn.execute('''
                        SELECT key, query, model_id, created_at, expires_at, hit_count
                        FROM ai_cache 
                        WHERE model_id = ? AND expires_at > ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (model_id, datetime.now(), limit))
                else:
                    cursor = self.conn.execute('''
                        SELECT key, query, model_id, created_at, expires_at, hit_count
                        FROM ai_cache 
                        WHERE expires_at > ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (datetime.now(), limit))
                
                for row in cursor.fetchall():
                    entries.append({
                        'key': row[0],
                        'query': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
                        'model_id': row[2],
                        'created_at': row[3],
                        'expires_at': row[4],
                        'hit_count': row[5]
                    })
            except Exception as e:
                logger.error(f"Error getting cache entries: {e}")
        
        return entries

# Global cache instance
cache_manager = None

def get_cache_manager() -> AICacheManager:
    """Get or create global cache manager instance"""
    global cache_manager
    if cache_manager is None:
        cache_type = os.getenv("AI_CACHE_TYPE", "sqlite")
        ttl_seconds = int(os.getenv("AI_CACHE_TTL", "3600"))
        max_size = int(os.getenv("AI_CACHE_MAX_SIZE", "10000"))
        
        cache_manager = AICacheManager(
            cache_type=cache_type,
            ttl_seconds=ttl_seconds,
            max_size=max_size
        )
    
    return cache_manager