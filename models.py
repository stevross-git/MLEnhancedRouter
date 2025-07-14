from app import db
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON

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
