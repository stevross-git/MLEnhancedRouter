import os
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Optional
import threading
from ml_router import MLEnhancedQueryRouter
from config import EnhancedRouterConfig
from model_manager import ModelManager, ModelType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///query_router.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions
db.init_app(app)

# Simple rate limiting dictionary
rate_limits = {}

# Global router instance
router = None
router_config = None
model_manager = None

def initialize_router():
    """Initialize the ML router in a background thread"""
    global router, router_config, model_manager
    
    try:
        router_config = EnhancedRouterConfig.from_env()
        model_manager = ModelManager()
        router = MLEnhancedQueryRouter(router_config, model_manager)
        
        # Initialize ML models
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(router.initialize())
        
        logger.info("ML Router initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML Router: {e}")
        router = None

# Initialize router in background thread
threading.Thread(target=initialize_router, daemon=True).start()

@app.route('/')
def index():
    """Main page with query submission form"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard showing routing statistics"""
    return render_template('dashboard.html')

@app.route('/agents')
def agents():
    """Agent management page"""
    return render_template('agents.html')

@app.route('/models')
def models():
    """Model management page"""
    return render_template('models.html')

@app.route('/api/query', methods=['POST'])
def submit_query():
    """Submit a query for routing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        user_id = data.get('user_id', 'anonymous')
        
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        # Process query asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(router.route_query(query, user_id))
            return jsonify(result)
        except Exception as e:
            logger.error(f"Query routing error: {e}")
            return jsonify({'error': 'Failed to process query'}), 500
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of available agents"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        agents = []
        for agent_id, agent in router.agents.items():
            agents.append({
                'id': agent_id,
                'name': agent.name,
                'description': agent.description,
                'categories': [cat.value for cat in agent.categories],
                'load_factor': agent.load_factor,
                'is_healthy': agent.is_healthy,
                'last_health_check': agent.last_health_check.isoformat() if agent.last_health_check else None
            })
        
        return jsonify({'agents': agents})
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        return jsonify({'error': 'Failed to get agents'}), 500

@app.route('/api/agents/register', methods=['POST'])
def register_agent():
    """Register a new agent"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Agent data is required'}), 400
        
        required_fields = ['name', 'description', 'categories', 'endpoint']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        # Register agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            agent_id = loop.run_until_complete(router.register_agent(
                name=data['name'],
                description=data['description'],
                categories=data['categories'],
                endpoint=data['endpoint'],
                capabilities=data.get('capabilities', {}),
                meta_data=data.get('meta_data', {})
            ))
            
            return jsonify({'agent_id': agent_id, 'message': 'Agent registered successfully'})
        except Exception as e:
            logger.error(f"Agent registration error: {e}")
            return jsonify({'error': 'Failed to register agent'}), 500
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/agents/<agent_id>', methods=['DELETE'])
def unregister_agent(agent_id):
    """Unregister an agent"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        if agent_id in router.agents:
            del router.agents[agent_id]
            return jsonify({'message': 'Agent unregistered successfully'})
        else:
            return jsonify({'error': 'Agent not found'}), 404
            
    except Exception as e:
        logger.error(f"Error unregistering agent: {e}")
        return jsonify({'error': 'Failed to unregister agent'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get routing statistics"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        stats = {
            'total_queries': router.total_queries,
            'successful_routes': router.successful_routes,
            'failed_routes': router.failed_routes,
            'cache_hits': router.cache_hits,
            'cache_misses': router.cache_misses,
            'active_agents': len([a for a in router.agents.values() if a.is_healthy]),
            'total_agents': len(router.agents),
            'avg_response_time': router.avg_response_time,
            'category_distribution': router.category_stats
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy' if router else 'unhealthy',
            'router_initialized': router is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if router:
            status['ml_classifier_initialized'] = router.ml_classifier.initialized
            status['agents_count'] = len(router.agents)
            
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Model Management API Endpoints
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all models"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_all_models()
        return jsonify({
            'models': [model.to_dict() for model in models]
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to get models'}), 500

@app.route('/api/models', methods=['POST'])
def create_model():
    """Create a new model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Model data is required'}), 400
        
        required_fields = ['name', 'description', 'type', 'categories']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        model_type = ModelType(data['type'])
        model = model_manager.create_model(
            name=data['name'],
            description=data['description'],
            model_type=model_type,
            categories=data['categories'],
            config=data.get('config', {})
        )
        
        return jsonify({
            'model_id': model.id,
            'message': 'Model created successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({'error': 'Failed to create model'}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get a specific model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'model': model.to_dict()})
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        return jsonify({'error': 'Failed to get model'}), 500

@app.route('/api/models/<model_id>', methods=['PUT'])
def update_model(model_id):
    """Update a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Model data is required'}), 400
        
        model_type = ModelType(data['type']) if 'type' in data else None
        
        model = model_manager.update_model(
            model_id=model_id,
            name=data.get('name'),
            description=data.get('description'),
            model_type=model_type,
            config=data.get('config')
        )
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'message': 'Model updated successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({'error': 'Failed to update model'}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.delete_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found or cannot be deleted'}), 404
        
        return jsonify({'message': 'Model deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

@app.route('/api/models/<model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """Activate a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.activate_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'message': 'Model activated successfully'})
        
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({'error': 'Failed to activate model'}), 500

@app.route('/api/models/<model_id>/train', methods=['POST'])
def train_model(model_id):
    """Train a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.train_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'message': 'Model training started'})
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'error': 'Failed to train model'}), 500

@app.route('/api/models/stats', methods=['GET'])
def get_model_stats():
    """Get model statistics"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        stats = model_manager.get_model_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        return jsonify({'error': 'Failed to get model stats'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

# Initialize database
with app.app_context():
    import models
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
