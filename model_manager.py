#!/usr/bin/env python3
"""
Model Manager for ML Query Router
Handles creation, training, and management of classification models
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class ModelType(Enum):
    KEYWORD = "keyword"
    RULE = "rule"
    HYBRID = "hybrid"

class ModelStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"

@dataclass
class MLModel:
    """Represents a machine learning model for query classification"""
    id: str
    name: str
    description: str
    type: ModelType
    categories: List[str]
    config: Dict[str, Any]
    status: ModelStatus = ModelStatus.INACTIVE
    accuracy: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'type': self.type.value,
            'categories': self.categories,
            'config': self.config,
            'status': self.status.value,
            'accuracy': self.accuracy,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'is_active': self.is_active
        }

class ModelManager:
    """Manages ML models for query classification"""
    
    def __init__(self, db=None):
        self.db = db
        self.models: Dict[str, MLModel] = {}
        self.active_model_id: Optional[str] = None
        self._load_models_from_db()
        if not self.models:
            self._initialize_default_models()
    
    def _load_models_from_db(self):
        """Load models from database"""
        if not self.db:
            return
            
        try:
            from models import MLModelRegistry
            
            db_models = MLModelRegistry.query.all()
            
            for db_model in db_models:
                model = MLModel(
                    id=db_model.id,
                    name=db_model.name,
                    description=db_model.description or "",
                    type=ModelType(db_model.model_type),
                    categories=db_model.categories,
                    config=db_model.config,
                    status=ModelStatus(db_model.status),
                    accuracy=db_model.accuracy,
                    created_at=db_model.created_at,
                    updated_at=db_model.updated_at,
                    version=db_model.version,
                    is_active=db_model.is_active
                )
                
                self.models[model.id] = model
                if model.is_active:
                    self.active_model_id = model.id
            
            logger.info(f"Loaded {len(db_models)} models from database")
            
        except Exception as e:
            logger.error(f"Failed to load models from database: {e}")

    def _save_model_to_db(self, model: MLModel):
        """Save model to database"""
        if not self.db:
            return
            
        try:
            from models import MLModelRegistry
            
            db_model = MLModelRegistry.query.get(model.id)
            
            if db_model:
                # Update existing model
                db_model.name = model.name
                db_model.description = model.description
                db_model.model_type = model.type.value
                db_model.categories = model.categories
                db_model.config = model.config
                db_model.status = model.status.value
                db_model.accuracy = model.accuracy
                db_model.updated_at = model.updated_at
                db_model.version = model.version
                db_model.is_active = model.is_active
            else:
                # Create new model
                db_model = MLModelRegistry(
                    id=model.id,
                    name=model.name,
                    description=model.description,
                    model_type=model.type.value,
                    categories=model.categories,
                    config=model.config,
                    status=model.status.value,
                    accuracy=model.accuracy,
                    is_active=model.is_active,
                    version=model.version,
                    created_at=model.created_at,
                    updated_at=model.updated_at
                )
                self.db.session.add(db_model)
            
            self.db.session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save model to database: {e}")
            self.db.session.rollback()

    def _delete_model_from_db(self, model_id: str):
        """Delete model from database"""
        if not self.db:
            return
            
        try:
            from models import MLModelRegistry
            
            db_model = MLModelRegistry.query.get(model_id)
            if db_model:
                self.db.session.delete(db_model)
                self.db.session.commit()
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete model from database: {e}")
            self.db.session.rollback()

    def _initialize_default_models(self):
        """Initialize default models"""
        try:
            # Create default keyword-based model
            default_model = self.create_model(
                name="Default Keyword Model",
                description="Default keyword-based classification model",
                model_type=ModelType.KEYWORD,
                categories=[
                    "analysis", "creative", "technical", "mathematical", 
                    "coding", "research", "philosophical", "practical", 
                    "educational", "conversational"
                ],
                config={
                    "keywords": {
                        "analysis": ["analyze", "examine", "investigate", "pattern", "trend", "data", "insight", "metrics"],
                        "creative": ["create", "write", "imagine", "design", "story", "poem", "generate", "invent"],
                        "technical": ["technical", "system", "architecture", "infrastructure", "deploy", "configure"],
                        "mathematical": ["calculate", "solve", "equation", "formula", "math", "compute", "derivative"],
                        "coding": ["code", "program", "function", "debug", "implement", "algorithm", "script"],
                        "research": ["research", "study", "find", "discover", "investigate", "source", "literature"],
                        "philosophical": ["meaning", "philosophy", "ethics", "moral", "existence", "purpose", "why"],
                        "practical": ["how to", "guide", "steps", "practical", "apply", "use", "tutorial"],
                        "educational": ["explain", "teach", "learn", "understand", "what is", "define", "describe"],
                        "conversational": ["chat", "talk", "discuss", "opinion", "think", "feel", "believe"]
                    }
                }
            )
            
            # Set as active model
            self.activate_model(default_model.id)
            
            logger.info(f"Default model initialized: {default_model.id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize default models: {e}")
    
    def create_model(self, name: str, description: str, model_type: ModelType, 
                    categories: List[str], config: Dict[str, Any]) -> MLModel:
        """Create a new model"""
        model_id = str(uuid.uuid4())
        
        model = MLModel(
            id=model_id,
            name=name,
            description=description,
            type=model_type,
            categories=categories,
            config=config,
            status=ModelStatus.INACTIVE
        )
        
        self.models[model_id] = model
        self._save_model_to_db(model)
        logger.info(f"Created model: {name} ({model_id})")
        
        return model
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get a model by ID"""
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[MLModel]:
        """Get all models"""
        return list(self.models.values())
    
    def get_active_model(self) -> Optional[MLModel]:
        """Get the currently active model"""
        if self.active_model_id:
            return self.models.get(self.active_model_id)
        return None
    
    def update_model(self, model_id: str, name: str = None, description: str = None,
                    model_type: ModelType = None, config: Dict[str, Any] = None) -> Optional[MLModel]:
        """Update an existing model"""
        model = self.models.get(model_id)
        if not model:
            return None
        
        if name:
            model.name = name
        if description:
            model.description = description
        if model_type:
            model.type = model_type
        if config:
            model.config = config
        
        model.updated_at = datetime.now()
        model.version += 1
        
        self._save_model_to_db(model)
        logger.info(f"Updated model: {model.name} ({model_id})")
        return model
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        if model_id in self.models:
            model = self.models[model_id]
            
            # Don't delete if it's the active model
            if model_id == self.active_model_id:
                return False
            
            del self.models[model_id]
            self._delete_model_from_db(model_id)
            logger.info(f"Deleted model: {model.name} ({model_id})")
            return True
        return False
    
    def activate_model(self, model_id: str) -> bool:
        """Activate a model"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        # Deactivate current active model
        if self.active_model_id:
            current_model = self.models.get(self.active_model_id)
            if current_model:
                current_model.is_active = False
                current_model.status = ModelStatus.INACTIVE
                self._save_model_to_db(current_model)
        
        # Activate new model
        model.is_active = True
        model.status = ModelStatus.ACTIVE
        self.active_model_id = model_id
        self._save_model_to_db(model)
        
        logger.info(f"Activated model: {model.name} ({model_id})")
        return True
    
    def train_model(self, model_id: str) -> bool:
        """Train a model (simulation for now)"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        model.status = ModelStatus.TRAINING
        logger.info(f"Started training model: {model.name} ({model_id})")
        
        # Simulate training completion
        import threading
        import time
        
        def complete_training():
            time.sleep(2)  # Simulate training time
            model.status = ModelStatus.INACTIVE
            model.accuracy = 0.85 + (hash(model_id) % 100) / 1000  # Simulate accuracy
            logger.info(f"Completed training model: {model.name} ({model_id})")
        
        thread = threading.Thread(target=complete_training)
        thread.daemon = True
        thread.start()
        
        return True
    
    def classify_query(self, query: str) -> tuple:
        """Classify a query using the active model"""
        active_model = self.get_active_model()
        if not active_model:
            return "conversational", 0.5
        
        return self._classify_with_model(query, active_model)
    
    def _classify_with_model(self, query: str, model: MLModel) -> tuple:
        """Classify query with a specific model"""
        if model.type == ModelType.KEYWORD:
            return self._classify_keyword(query, model)
        elif model.type == ModelType.RULE:
            return self._classify_rule(query, model)
        elif model.type == ModelType.HYBRID:
            return self._classify_hybrid(query, model)
        else:
            return "conversational", 0.5
    
    def _classify_keyword(self, query: str, model: MLModel) -> tuple:
        """Keyword-based classification"""
        query_lower = query.lower()
        category_scores = {}
        
        keywords = model.config.get('keywords', {})
        
        for category, keyword_list in keywords.items():
            if category in model.categories:
                score = 0
                for keyword in keyword_list:
                    if keyword in query_lower:
                        position = query_lower.find(keyword)
                        weight = 1.0 - (position / len(query_lower)) * 0.5
                        score += weight
                
                if score > 0:
                    category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            total_score = sum(category_scores.values())
            confidence = min(best_category[1] / total_score if total_score > 0 else 0.5, 1.0)
            return best_category[0], confidence
        else:
            return "conversational", 0.5
    
    def _classify_rule(self, query: str, model: MLModel) -> tuple:
        """Rule-based classification"""
        rules = model.config.get('rules', [])
        
        # Simple rule evaluation (can be extended)
        query_lower = query.lower()
        
        for rule in rules:
            if isinstance(rule, str):
                if rule.lower() in query_lower:
                    # Extract category from rule (simplified)
                    for category in model.categories:
                        if category in rule.lower():
                            return category, 0.8
        
        return "conversational", 0.5
    
    def _classify_hybrid(self, query: str, model: MLModel) -> tuple:
        """Hybrid classification (combination of keyword and rule)"""
        # Try keyword first
        keyword_result = self._classify_keyword(query, model)
        
        # If confidence is low, try rules
        if keyword_result[1] < 0.6:
            rule_result = self._classify_rule(query, model)
            if rule_result[1] > keyword_result[1]:
                return rule_result
        
        return keyword_result
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about models"""
        total_models = len(self.models)
        active_models = sum(1 for m in self.models.values() if m.is_active)
        training_models = sum(1 for m in self.models.values() if m.status == ModelStatus.TRAINING)
        
        accuracies = [m.accuracy for m in self.models.values() if m.accuracy is not None]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        return {
            'total_models': total_models,
            'active_models': active_models,
            'training_models': training_models,
            'avg_accuracy': avg_accuracy
        }