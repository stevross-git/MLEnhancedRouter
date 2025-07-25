import os
import logging
import time
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response

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
from ai_models import AIModelManager, AIProvider
from auth_system import AuthManager, UserRole
from ai_cache import get_cache_manager
from rag_chat import get_rag_chat
from swagger_spec import swagger_spec
from collaborative_router import get_collaborative_router
from shared_memory import get_shared_memory_manager
from external_llm_integration import get_external_llm_manager

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

# Global instances
router = None
router_config = None
model_manager = None
ai_model_manager = None
auth_manager = None
cache_manager = None
rag_system = None
collaborative_router = None
shared_memory_manager = None
external_llm_manager = None

def initialize_router():
    """Initialize the ML router in a background thread"""
    global router, router_config, model_manager, ai_model_manager, auth_manager, cache_manager, rag_system, collaborative_router, shared_memory_manager, external_llm_manager
    
    try:
        with app.app_context():
            router_config = EnhancedRouterConfig.from_env()
            model_manager = ModelManager(db)
            ai_model_manager = AIModelManager(db)
            auth_manager = AuthManager()
            cache_manager = get_cache_manager(db)
            rag_system = get_rag_chat()
            shared_memory_manager = get_shared_memory_manager()
            external_llm_manager = get_external_llm_manager(db)
            router = MLEnhancedQueryRouter(router_config, model_manager)
            collaborative_router = get_collaborative_router(ai_model_manager)
            
            # Initialize ML models
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(router.initialize())
            
            logger.info("ML Router initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML Router: {e}")
        router = None

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

@app.route('/ai-models')
def ai_models():
    """AI model management page"""
    return render_template('ai_models.html')

@app.route('/chat')
def chat():
    """Advanced chat interface"""
    return render_template('chat.html')

@app.route('/auth')
def auth():
    """Authentication management page"""
    return render_template('auth.html')

@app.route('/settings')
def settings():
    """Settings page for API key management"""
    return render_template('settings.html')

@app.route('/config')
def config():
    """Configuration page for advanced settings"""
    return render_template('config.html')

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

# AI Models API Routes
@app.route('/api/ai-models', methods=['GET'])
def get_ai_models():
    """Get all AI models"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        models = ai_model_manager.get_all_models()
        models_data = []
        for model in models:
            models_data.append({
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value,
                'model_name': model.model_name,
                'endpoint': model.endpoint,
                'api_key_env': model.api_key_env,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'top_p': model.top_p,
                'context_window': model.context_window,
                'cost_per_1k_tokens': model.cost_per_1k_tokens,
                'is_active': model.is_active,
                'supports_streaming': model.supports_streaming,
                'supports_system_message': model.supports_system_message
            })
        
        return jsonify({'status': 'success', 'models': models_data})
        
    except Exception as e:
        logger.error(f"Error getting AI models: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get AI models'}), 500

@app.route('/api/ai-models', methods=['POST'])
def create_ai_model():
    """Create a new AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        data = request.get_json()
        required_fields = ['id', 'name', 'provider', 'model_name', 'endpoint']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        model = ai_model_manager.add_custom_model(
            model_id=data['id'],
            name=data['name'],
            endpoint=data['endpoint'],
            api_key_env=data.get('api_key_env', ''),
            model_name=data['model_name'],
            max_tokens=data.get('max_tokens', 4096),
            temperature=data.get('temperature', 0.7),
            custom_headers=data.get('custom_headers', {})
        )
        
        return jsonify({
            'status': 'success',
            'message': 'AI model created successfully',
            'model_id': model.id
        })
        
    except Exception as e:
        logger.error(f"Error creating AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to create AI model'}), 500

@app.route('/api/ai-models/<model_id>', methods=['DELETE'])
def delete_ai_model(model_id):
    """Delete an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        success = ai_model_manager.remove_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'status': 'success', 'message': 'AI model deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to delete AI model'}), 500

@app.route('/api/ai-models/activate/<model_id>', methods=['POST'])
def activate_ai_model(model_id):
    """Activate an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        success = ai_model_manager.set_active_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'status': 'success', 'message': 'AI model activated successfully'})
        
    except Exception as e:
        logger.error(f"Error activating AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to activate AI model'}), 500

@app.route('/api/ai-models/active', methods=['GET'])
def get_active_ai_model():
    """Get the active AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        active_model = ai_model_manager.get_active_model()
        if not active_model:
            return jsonify({'status': 'success', 'model': None})
        
        model_data = {
            'id': active_model.id,
            'name': active_model.name,
            'provider': active_model.provider.value,
            'model_name': active_model.model_name,
            'endpoint': active_model.endpoint,
            'api_key_env': active_model.api_key_env,
            'max_tokens': active_model.max_tokens,
            'temperature': active_model.temperature,
            'top_p': active_model.top_p,
            'context_window': active_model.context_window,
            'cost_per_1k_tokens': active_model.cost_per_1k_tokens,
            'is_active': active_model.is_active,
            'supports_streaming': active_model.supports_streaming,
            'supports_system_message': active_model.supports_system_message
        }
        
        return jsonify({'status': 'success', 'model': model_data})
        
    except Exception as e:
        logger.error(f"Error getting active AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get active AI model'}), 500

@app.route('/api/ai-models/test/<model_id>', methods=['POST'])
def test_ai_model(model_id):
    """Test an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        data = request.get_json()
        query = data.get('query', 'Hello! Can you confirm you are working correctly?')
        
        # Test the model
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(ai_model_manager.generate_response(
                query=query,
                model_id=model_id
            ))
            return jsonify(result)
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error testing AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to test AI model'}), 500

@app.route('/api/ai-models/api-key-status', methods=['GET'])
def get_api_key_status():
    """Get API key status for all providers"""
    try:
        providers = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GEMINI_API_KEY',
            'xai': 'XAI_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY'
        }
        
        status_info = {}
        for provider, env_var in providers.items():
            api_key = os.getenv(env_var)
            status_info[provider] = {
                'available': bool(api_key),
                'message': 'API key configured' if api_key else 'API key not configured'
            }
        
        # Ollama is always available (local)
        status_info['ollama'] = {
            'available': True,
            'message': 'Local endpoint'
        }
        
        return jsonify({'status': 'success', 'status_info': status_info})
        
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get API key status'}), 500

# Authentication API Routes
@app.route('/api/auth/current-user', methods=['GET'])
def get_current_user():
    """Get current user info"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        # For now, return the admin user
        admin_user = auth_manager.users.get('admin')
        if not admin_user:
            return jsonify({'status': 'error', 'error': 'No user found'}), 404
        
        user_data = {
            'id': admin_user.id,
            'username': admin_user.username,
            'email': admin_user.email,
            'role': admin_user.role.value,
            'api_key': admin_user.api_key,
            'created_at': admin_user.created_at.isoformat(),
            'last_login': admin_user.last_login.isoformat() if admin_user.last_login else None,
            'is_active': admin_user.is_active,
            'permissions': admin_user.permissions
        }
        
        return jsonify({'status': 'success', 'user': user_data})
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get current user'}), 500

@app.route('/api/auth/users', methods=['GET'])
def get_all_users():
    """Get all users"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        users = auth_manager.get_all_users()
        users_data = []
        
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'is_active': user.is_active
            })
        
        return jsonify({'status': 'success', 'users': users_data})
        
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get users'}), 500

@app.route('/api/auth/regenerate-api-key', methods=['POST'])
def regenerate_api_key():
    """Regenerate API key for current user"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        # For now, regenerate for admin user
        new_api_key = auth_manager.regenerate_api_key('admin')
        if not new_api_key:
            return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500
        
        return jsonify({'status': 'success', 'api_key': new_api_key})
        
    except Exception as e:
        logger.error(f"Error regenerating API key: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500

@app.route('/api/auth/generate-jwt', methods=['POST'])
def generate_jwt():
    """Generate JWT token"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        data = request.get_json()
        expires_in = data.get('expires_in', 3600)
        
        # For now, generate for admin user
        token = auth_manager.generate_jwt_token('admin', expires_in)
        
        return jsonify({'status': 'success', 'token': token})
        
    except Exception as e:
        logger.error(f"Error generating JWT: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to generate JWT'}), 500

@app.route('/api/auth/usage-stats', methods=['GET'])
def get_usage_stats():
    """Get API usage statistics"""
    try:
        # Return mock data for now
        stats = {
            'total_requests': 150,
            'requests_today': 25,
            'error_rate': 2.5
        }
        
        return jsonify({'status': 'success', 'stats': stats})
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get usage stats'}), 500

# Settings API Routes
@app.route('/api/settings/save-api-keys', methods=['POST'])
def save_api_keys():
    """Save API keys to environment"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # For demo purposes, we'll just validate the keys
        # In production, these would be securely stored
        saved_keys = {}
        key_mappings = {
            'openai-key': 'OPENAI_API_KEY',
            'anthropic-key': 'ANTHROPIC_API_KEY',
            'google-key': 'GEMINI_API_KEY',
            'xai-key': 'XAI_API_KEY',
            'perplexity-key': 'PERPLEXITY_API_KEY',
            'cohere-key': 'COHERE_API_KEY',
            'mistral-key': 'MISTRAL_API_KEY',
            'huggingface-key': 'HUGGINGFACE_API_KEY'
        }
        
        for form_key, env_key in key_mappings.items():
            if form_key in data and data[form_key]:
                saved_keys[env_key] = data[form_key]
                # In production, you would save to secure storage
                # os.environ[env_key] = data[form_key]
        
        return jsonify({
            'status': 'success',
            'message': f'Saved {len(saved_keys)} API keys',
            'saved_keys': list(saved_keys.keys())
        })
        
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to save API keys'}), 500

@app.route('/api/settings/general', methods=['GET', 'POST'])
def general_settings():
    """Get or save general settings"""
    try:
        if request.method == 'GET':
            # Return current settings
            settings = {
                'default_model': 'gpt-4o-mini',
                'max_tokens': 4096,
                'temperature': 0.7,
                'auto_retry': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Save settings (in production, save to database)
            return jsonify({
                'status': 'success',
                'message': 'General settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with general settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle general settings'}), 500

@app.route('/api/settings/security', methods=['GET', 'POST'])
def security_settings():
    """Get or save security settings"""
    try:
        if request.method == 'GET':
            settings = {
                'rate_limit': 60,
                'session_timeout': 60,
                'require_auth': True,
                'log_requests': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Security settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with security settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle security settings'}), 500

@app.route('/api/settings/performance', methods=['GET', 'POST'])
def performance_settings():
    """Get or save performance settings"""
    try:
        if request.method == 'GET':
            settings = {
                'cache_ttl': 3600,
                'max_concurrent': 10,
                'request_timeout': 30,
                'enable_cache': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Performance settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with performance settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle performance settings'}), 500

# Configuration API Routes
@app.route('/api/config/model', methods=['GET', 'POST'])
def model_config():
    """Get or save model configuration"""
    try:
        if request.method == 'GET':
            config = {
                'openai': {
                    'model': 'gpt-4o',
                    'max_tokens': 4096,
                    'temperature': 0.7
                },
                'anthropic': {
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 4096,
                    'temperature': 0.7
                },
                'google': {
                    'model': 'gemini-2.5-flash',
                    'max_tokens': 8192
                },
                'xai': {
                    'model': 'grok-2-1212',
                    'max_tokens': 131072
                },
                'ollama': {
                    'endpoint': 'http://localhost:11434',
                    'model': 'llama3.2'
                }
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Model configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with model config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle model configuration'}), 500

@app.route('/api/config/endpoint', methods=['GET', 'POST'])
def endpoint_config():
    """Get or save endpoint configuration"""
    try:
        if request.method == 'GET':
            config = {
                'openai_endpoint': 'https://api.openai.com/v1',
                'anthropic_endpoint': 'https://api.anthropic.com',
                'google_endpoint': 'https://generativelanguage.googleapis.com/v1beta',
                'xai_endpoint': 'https://api.x.ai/v1',
                'connection_timeout': 30,
                'read_timeout': 60,
                'retry_attempts': 3,
                'retry_delay': 1,
                'custom_endpoints': []
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Endpoint configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with endpoint config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle endpoint configuration'}), 500

@app.route('/api/config/routing', methods=['GET', 'POST'])
def routing_config():
    """Get or save routing configuration"""
    try:
        if request.method == 'GET':
            config = {
                'confidence_threshold': 0.7,
                'fallback_strategy': 'keyword',
                'enable_ml_classification': True,
                'load_balancing': 'least-connections',
                'max_agents': 5,
                'agent_timeout': 30,
                'enabled_categories': [
                    'analysis', 'creative', 'technical', 'coding',
                    'mathematical', 'research', 'philosophical',
                    'practical', 'educational', 'conversational'
                ]
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Routing configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with routing config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle routing configuration'}), 500

@app.route('/api/config/monitoring', methods=['GET', 'POST'])
def monitoring_config():
    """Get or save monitoring configuration"""
    try:
        if request.method == 'GET':
            config = {
                'log_level': 'INFO',
                'log_retention': 30,
                'log_queries': True,
                'log_responses': True,
                'metrics_endpoint': 'http://localhost:9090',
                'metrics_interval': 60,
                'enable_metrics': True,
                'enable_health_checks': True,
                'error_threshold': 5,
                'response_threshold': 5000,
                'alert_email': ''
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Monitoring configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with monitoring config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle monitoring configuration'}), 500

@app.route('/api/config/advanced', methods=['GET', 'POST'])
def advanced_config():
    """Get or save advanced configuration"""
    try:
        if request.method == 'GET':
            config = {
                'thread_pool_size': 10,
                'connection_pool_size': 20,
                'queue_size': 1000,
                'cache_backend': 'redis',
                'cache_ttl': 3600,
                'cache_max_size': 1024,
                'feature_flags': {
                    'enable_streaming': True,
                    'enable_caching': True,
                    'enable_compression': True,
                    'enable_rate_limiting': True,
                    'enable_circuit_breaker': False,
                    'enable_distributed_tracing': False,
                    'enable_auto_scaling': False,
                    'enable_backup': False,
                    'debug_mode': False
                }
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Advanced configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with advanced config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle advanced configuration'}), 500

@app.route('/api/config/export', methods=['GET'])
def export_config():
    """Export complete configuration"""
    try:
        # In production, this would gather all actual configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'model_config': {},
            'endpoint_config': {},
            'routing_config': {},
            'monitoring_config': {},
            'advanced_config': {}
        }
        
        return jsonify({'status': 'success', 'config': config})
        
    except Exception as e:
        logger.error(f"Error exporting config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to export configuration'}), 500

@app.route('/api/config/import', methods=['POST'])
def import_config():
    """Import configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # In production, this would validate and apply the configuration
        return jsonify({
            'status': 'success',
            'message': 'Configuration imported successfully'
        })
        
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to import configuration'}), 500

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

# Cache Management Endpoints
@app.route('/api/cache/stats')
def get_cache_stats():
    """Get cache statistics"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        stats = cache_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/entries')
def get_cache_entries():
    """Get cache entries"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        model_id = request.args.get('model_id')
        limit = int(request.args.get('limit', 100))
        
        entries = cache_manager.get_cache_entries(model_id=model_id, limit=limit)
        return jsonify(entries)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache entries"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id')
        
        cache_manager.clear(model_id=model_id)
        return jsonify({'status': 'success', 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chat API Endpoints
@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    """Send message to AI model"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        system_message = data.get('system_message')
        model_id = data.get('model_id')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 4096)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 500
        
        # Generate response using AI model manager
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                ai_model_manager.generate_response(
                    query=query,
                    system_message=system_message,
                    model_id=model_id,
                    user_id=session.get('user_id', 'anonymous')
                )
            )
            
            return jsonify({
                'status': 'success',
                'response': response['response'],
                'model': response['model'],
                'usage': response.get('usage', {}),
                'cached': response.get('cached', False)
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream')
def chat_stream():
    """Stream chat response using Server-Sent Events"""
    query = request.args.get('query', '')
    system_message = request.args.get('system_message')
    model_id = request.args.get('model_id')
    
    if not query or not model_id:
        return jsonify({'error': 'Query and model_id are required'}), 400
    
    def generate():
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': model_id})}\n\n"
            
            # For now, simulate streaming by chunking the response
            # In a real implementation, you'd integrate with streaming APIs
            
            # Get regular response first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    ai_model_manager.generate_response(
                        query=query,
                        system_message=system_message,
                        model_id=model_id,
                        user_id=session.get('user_id', 'anonymous')
                    )
                )
                
                # Simulate streaming by sending chunks
                full_response = response['response']
                words = full_response.split()
                
                for i, word in enumerate(words):
                    chunk = word + (' ' if i < len(words) - 1 else '')
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    time.sleep(0.1)  # Simulate streaming delay
                
                yield f"data: {json.dumps({'type': 'end', 'usage': response.get('usage', {})})}\n\n"
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get user's chat sessions"""
    user_id = session.get('user_id', 'anonymous')
    
    # In a real implementation, you'd fetch from database
    # For now, return empty array as sessions are stored client-side
    return jsonify([])

@app.route('/api/chat/sessions', methods=['POST'])
def create_chat_session():
    """Create new chat session"""
    try:
        data = request.get_json()
        user_id = session.get('user_id', 'anonymous')
        
        # In a real implementation, you'd save to database
        session_data = {
            'id': f"chat_{int(time.time())}_{user_id}",
            'title': data.get('title', 'New Chat'),
            'model': data.get('model'),
            'created': datetime.now().isoformat(),
            'messages': []
        }
        
        return jsonify(session_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# RAG System Endpoints
@app.route('/api/rag/upload', methods=['POST'])
def upload_document():
    """Upload a document for RAG processing"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        temp_path = f"./temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Process the uploaded file
            doc_id = rag_system.process_uploaded_file(temp_path, file.filename)
            
            if doc_id:
                return jsonify({
                    'message': 'Document uploaded successfully',
                    'document_id': doc_id,
                    'filename': file.filename
                })
            else:
                return jsonify({'error': 'Failed to process document'}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        documents = rag_system.get_documents_list()
        return jsonify({'documents': documents})
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from RAG system"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        success = rag_system.delete_document(doc_id)
        if success:
            return jsonify({'message': 'Document deleted successfully'})
        else:
            return jsonify({'error': 'Document not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/search', methods=['POST'])
def search_documents():
    """Search documents using RAG system"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        data = request.get_json()
        query = data.get('query', '')
        max_results = data.get('max_results', 3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = rag_system.search_documents(query, max_results)
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG system statistics"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        stats = rag_system.get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return jsonify({'error': str(e)}), 500

# Swagger Documentation Endpoints
@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Interactive API documentation"""
    return render_template('api_docs.html')

@app.route('/api/openapi.json', methods=['GET'])
def openapi_spec():
    """OpenAPI/Swagger specification"""
    return jsonify(swagger_spec)

# External LLM Integration Endpoints
@app.route('/api/external-llm/analyze', methods=['POST'])
def analyze_query_complexity():
    """Analyze query complexity for external LLM routing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        complex_query = external_llm_manager.analyzer.analyze_query(query)
        
        return jsonify({
            'query': query,
            'complexity': complex_query.complexity.value,
            'domain': complex_query.domain,
            'requires_reasoning': complex_query.requires_reasoning,
            'requires_creativity': complex_query.requires_creativity,
            'requires_analysis': complex_query.requires_analysis,
            'requires_multi_step': complex_query.requires_multi_step,
            'context_length': complex_query.context_length,
            'specialized_knowledge': complex_query.specialized_knowledge,
            'is_complex': external_llm_manager.is_complex_query(query),
            'recommended_provider': external_llm_manager.get_recommended_provider(query).value if external_llm_manager.get_recommended_provider(query) else None
        })
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/process', methods=['POST'])
def process_with_external_llm():
    """Process a query using external LLM"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        context = data.get('context', '')
        preferred_provider = data.get('preferred_provider')
        
        # Convert string provider to enum if provided
        if preferred_provider:
            from external_llm_integration import ExternalProvider
            try:
                preferred_provider = ExternalProvider(preferred_provider)
            except ValueError:
                return jsonify({'error': f'Invalid provider: {preferred_provider}'}), 400
        
        # Process with external LLM
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            external_llm_manager.process_complex_query(
                query, context=context, preferred_provider=preferred_provider
            )
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing with external LLM: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/providers')
def get_external_providers():
    """Get available external LLM providers"""
    try:
        providers = []
        for provider_enum, config in external_llm_manager.providers.items():
            api_key = os.environ.get(config.api_key_env)
            providers.append({
                'id': provider_enum.value,
                'name': config.model_name,
                'endpoint': config.endpoint,
                'max_tokens': config.max_tokens,
                'cost_per_1k_tokens': config.cost_per_1k_tokens,
                'rate_limit_rpm': config.rate_limit_rpm,
                'specializations': config.specializations,
                'api_key_available': bool(api_key)
            })
        
        return jsonify({
            'providers': providers,
            'total_providers': len(providers),
            'available_providers': len([p for p in providers if p['api_key_available']])
        })
    except Exception as e:
        logger.error(f"Error getting external providers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/metrics')
def get_external_llm_metrics():
    """Get performance metrics for external LLM providers"""
    try:
        metrics = external_llm_manager.get_provider_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting external LLM metrics: {e}")
        return jsonify({'error': str(e)}), 500

# Collaborative AI Endpoints
@app.route('/api/collaborate', methods=['POST'])
def collaborate():
    """Submit a query for collaborative AI processing"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        enable_rag = data.get('enable_rag', False)
        max_agents = data.get('max_agents', 3)
        collaboration_timeout = data.get('timeout', 300)
        
        # Run collaborative processing
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                collaborative_router.process_collaborative_query(
                    query=query,
                    enable_rag=enable_rag,
                    max_agents=max_agents,
                    collaboration_timeout=collaboration_timeout
                )
            )
            return jsonify(result)
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Collaborative processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/sessions', methods=['GET'])
def get_collaboration_sessions():
    """Get active collaboration sessions"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        active_sessions = collaborative_router.get_active_sessions()
        return jsonify({'sessions': active_sessions})
        
    except Exception as e:
        logger.error(f"Error getting collaboration sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/sessions/<session_id>', methods=['GET'])
def get_collaboration_session(session_id):
    """Get details of a specific collaboration session"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        session_details = collaborative_router.get_session_details(session_id)
        return jsonify(session_details)
        
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/stats', methods=['GET'])
def get_shared_memory_stats():
    """Get shared memory statistics"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        # Get basic stats
        stats = {
            'total_messages': len(shared_memory_manager.messages),
            'active_sessions': len(shared_memory_manager.sessions),
            'agent_contexts': len(shared_memory_manager.agent_contexts),
            'message_index_size': sum(len(msgs) for msgs in shared_memory_manager.message_index.values())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting shared memory stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """Get messages from a specific session"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        message_types = request.args.getlist('types')
        
        # Convert string types to MessageType enum
        from shared_memory import MessageType
        if message_types:
            try:
                message_types = [MessageType(t) for t in message_types]
            except ValueError:
                return jsonify({'error': 'Invalid message type'}), 400
        else:
            message_types = None
        
        messages = shared_memory_manager.get_session_messages(
            session_id, 
            message_types=message_types
        )
        
        # Limit results
        messages = messages[-limit:]
        
        return jsonify({
            'session_id': session_id,
            'messages': [msg.to_dict() for msg in messages]
        })
        
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/sessions/<session_id>/context', methods=['GET'])
def get_session_context(session_id):
    """Get shared context for a session"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        context = shared_memory_manager.get_shared_context(session_id)
        return jsonify(context)
        
    except Exception as e:
        logger.error(f"Error getting session context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/collaborate')
def collaborate_page():
    """Collaborative AI interface page"""
    return render_template('collaborate.html')

@app.route('/api/collaborate/agents', methods=['GET'])
def get_collaborative_agents():
    """Get collaborative agent configurations"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        configurations = collaborative_router.get_agent_configurations()
        return jsonify(configurations)
        
    except Exception as e:
        logger.error(f"Error getting collaborative agents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/agents/<agent_id>/model', methods=['PUT'])
def update_agent_model(agent_id):
    """Update AI model for a specific collaborative agent"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        data = request.get_json()
        if not data or 'model_id' not in data:
            return jsonify({'error': 'model_id is required'}), 400
        
        model_id = data['model_id']
        success = collaborative_router.update_agent_model(agent_id, model_id)
        
        if success:
            return jsonify({'message': f'Agent {agent_id} updated to use model {model_id}'})
        else:
            return jsonify({'error': 'Failed to update agent model'}), 400
        
    except Exception as e:
        logger.error(f"Error updating agent model: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize database
with app.app_context():
    import models
    db.create_all()
    
    # Initialize router after models are imported
    threading.Thread(target=initialize_router, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
