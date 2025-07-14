from app import db
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
import hashlib

class QueryLog(db.Model):
    """Log of all queries processed by the router"""
    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    user_id = Column(String(64), nullable=True)
    category = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    agent_id = Column(String(64), nullable=True)
    agent_name = Column(String(128), nullable=True)
    status = Column(String(32), nullable=False)
    response_time = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)

class AgentRegistration(db.Model):
    """Registry of all agents"""
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    endpoint = Column(String(256), nullable=False)
    categories = Column(JSON, nullable=False)
    capabilities = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)

class RouterMetrics(db.Model):
    """Performance metrics for the router"""
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)

class MLModelRegistry(db.Model):
    """Registry of ML models"""
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(32), nullable=False)
    categories = Column(JSON, nullable=False)
    config = Column(JSON, nullable=False)
    status = Column(String(32), default='inactive')
    accuracy = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class AICacheEntry(db.Model):
    """AI response cache entries"""
    __tablename__ = 'ai_cache_entries'
    
    id = Column(Integer, primary_key=True)
    cache_key = Column(String(64), unique=True, nullable=False, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_id = Column(String(64), nullable=False, index=True)
    system_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    meta_data = Column(JSON, nullable=True)
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    
    @classmethod
    def generate_cache_key(cls, query: str, model_id: str, system_message: str = None) -> str:
        """Generate a unique cache key for the query"""
        content = f"{query}|{model_id}|{system_message or ''}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def increment_hit_count(self):
        """Increment hit count and update last accessed time"""
        self.hit_count += 1
        self.last_accessed = datetime.utcnow()

class AICacheStats(db.Model):
    """AI cache statistics and metrics"""
    __tablename__ = 'ai_cache_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    total_requests = Column(Integer, default=0)
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    total_entries = Column(Integer, default=0)
    expired_entries = Column(Integer, default=0)
    cache_size_mb = Column(Float, default=0.0)
    average_response_time = Column(Float, default=0.0)
    model_id = Column(String(64), nullable=True, index=True)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100
    
    @property  
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_misses / self.total_requests) * 100

class ChatSession(db.Model):
    """Chat session model for storing conversation history"""
    __tablename__ = 'chat_sessions'
    
    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False, default='anonymous')
    title = Column(String(200), nullable=False)
    model_id = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Relationship to messages
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ChatSession {self.id}: {self.title}>'

class ChatMessage(db.Model):
    """Chat message model for storing individual messages"""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), db.ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    model_id = Column(String(100), nullable=True)
    system_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    usage_tokens = Column(Integer, default=0)
    cached = Column(Boolean, default=False)
    attachments = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f'<ChatMessage {self.id}: {self.role} - {self.content[:50]}...>'
