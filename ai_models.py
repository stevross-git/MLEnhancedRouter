"""
AI Model Integration Service
Supports major AI providers, local Ollama, and custom endpoints
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import aiohttp
import requests
from datetime import datetime
from ai_cache import get_cache_manager

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"

@dataclass
class AIModel:
    """AI model configuration"""
    id: str
    name: str
    provider: AIProvider
    model_name: str
    endpoint: str
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    supports_streaming: bool = True
    supports_system_message: bool = True
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4096
    is_active: bool = True
    custom_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}

class AIModelManager:
    """Manages AI model configurations and interactions"""
    
    def __init__(self, db=None):
        self.db = db
        self.models: Dict[str, AIModel] = {}
        self.active_model_id: Optional[str] = None
        self.cache_manager = get_cache_manager(db)
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default AI models"""
        default_models = [
            # OpenAI Models
            AIModel(
                id="gpt-4o",
                name="GPT-4o",
                provider=AIProvider.OPENAI,
                model_name="gpt-4o",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.03
            ),
            AIModel(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                provider=AIProvider.OPENAI,
                model_name="gpt-4-turbo",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.01
            ),
            AIModel(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                provider=AIProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                context_window=16385,
                cost_per_1k_tokens=0.0005
            ),
            # Anthropic Models
            AIModel(
                id="claude-sonnet-4",
                name="Claude Sonnet 4",
                provider=AIProvider.ANTHROPIC,
                model_name="claude-sonnet-4-20250514",
                endpoint="https://api.anthropic.com/v1/messages",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.015
            ),
            AIModel(
                id="claude-3-5-sonnet",
                name="Claude 3.5 Sonnet",
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                endpoint="https://api.anthropic.com/v1/messages",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=4096,
                context_window=200000,
                cost_per_1k_tokens=0.003
            ),
            # Google Models
            AIModel(
                id="gemini-2.5-flash",
                name="Gemini 2.5 Flash",
                provider=AIProvider.GOOGLE,
                model_name="gemini-2.5-flash",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=1048576,
                cost_per_1k_tokens=0.000125
            ),
            AIModel(
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                provider=AIProvider.GOOGLE,
                model_name="gemini-2.5-pro",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=2097152,
                cost_per_1k_tokens=0.00125
            ),
            # xAI Models
            AIModel(
                id="grok-2-1212",
                name="Grok 2 (1212)",
                provider=AIProvider.XAI,
                model_name="grok-2-1212",
                endpoint="https://api.x.ai/v1/chat/completions",
                api_key_env="XAI_API_KEY",
                max_tokens=4096,
                context_window=131072,
                cost_per_1k_tokens=0.002
            ),
            AIModel(
                id="grok-2-vision-1212",
                name="Grok 2 Vision (1212)",
                provider=AIProvider.XAI,
                model_name="grok-2-vision-1212",
                endpoint="https://api.x.ai/v1/chat/completions",
                api_key_env="XAI_API_KEY",
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.002
            ),
            # Perplexity Models
            AIModel(
                id="llama-3.1-sonar-small-128k-online",
                name="Llama 3.1 Sonar Small (Online)",
                provider=AIProvider.PERPLEXITY,
                model_name="llama-3.1-sonar-small-128k-online",
                endpoint="https://api.perplexity.ai/chat/completions",
                api_key_env="PERPLEXITY_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.0002
            ),
            AIModel(
                id="llama-3.1-sonar-large-128k-online",
                name="Llama 3.1 Sonar Large (Online)",
                provider=AIProvider.PERPLEXITY,
                model_name="llama-3.1-sonar-large-128k-online",
                endpoint="https://api.perplexity.ai/chat/completions",
                api_key_env="PERPLEXITY_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.001
            ),
            # Ollama (Local)
            AIModel(
                id="ollama-llama3",
                name="Llama 3 (Local)",
                provider=AIProvider.OLLAMA,
                model_name="llama3",
                endpoint="http://localhost:11434/api/chat",
                api_key_env="",
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.0
            ),
            AIModel(
                id="ollama-mistral",
                name="Mistral (Local)",
                provider=AIProvider.OLLAMA,
                model_name="mistral",
                endpoint="http://localhost:11434/api/chat",
                api_key_env="",
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.0
            ),
            # Cohere Models
            AIModel(
                id="command-r-plus",
                name="Command R+",
                provider=AIProvider.COHERE,
                model_name="command-r-plus",
                endpoint="https://api.cohere.ai/v1/chat",
                api_key_env="COHERE_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.003
            ),
            # Mistral Models
            AIModel(
                id="mistral-large-2407",
                name="Mistral Large 2407",
                provider=AIProvider.MISTRAL,
                model_name="mistral-large-2407",
                endpoint="https://api.mistral.ai/v1/chat/completions",
                api_key_env="MISTRAL_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.008
            ),
            AIModel(
                id="mistral-small-2409",
                name="Mistral Small 2409",
                provider=AIProvider.MISTRAL,
                model_name="mistral-small-2409",
                endpoint="https://api.mistral.ai/v1/chat/completions",
                api_key_env="MISTRAL_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.001
            )
        ]
        
        for model in default_models:
            self.models[model.id] = model
            
        # Set default active model
        if "gpt-4o" in self.models:
            self.active_model_id = "gpt-4o"
        elif self.models:
            self.active_model_id = next(iter(self.models.keys()))
    
    def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get a model by ID"""
        return self.models.get(model_id)
    
    def get_all_models(self) -> List[AIModel]:
        """Get all available models"""
        return list(self.models.values())
    
    def get_models_by_provider(self, provider: AIProvider) -> List[AIModel]:
        """Get models by provider"""
        return [model for model in self.models.values() if model.provider == provider]
    
    def get_active_model(self) -> Optional[AIModel]:
        """Get the currently active model"""
        return self.models.get(self.active_model_id) if self.active_model_id else None
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model"""
        if model_id in self.models:
            self.active_model_id = model_id
            return True
        return False
    
    def add_custom_model(self, model_id: str, name: str, endpoint: str, 
                        api_key_env: str, model_name: str = None,
                        max_tokens: int = 4096, temperature: float = 0.7,
                        custom_headers: Dict[str, str] = None) -> AIModel:
        """Add a custom model"""
        model = AIModel(
            id=model_id,
            name=name,
            provider=AIProvider.CUSTOM,
            model_name=model_name or model_id,
            endpoint=endpoint,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            custom_headers=custom_headers or {}
        )
        
        self.models[model_id] = model
        return model
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model"""
        if model_id in self.models:
            del self.models[model_id]
            if self.active_model_id == model_id:
                self.active_model_id = next(iter(self.models.keys())) if self.models else None
            return True
        return False
    
    async def generate_response(self, query: str, system_message: str = None, 
                              model_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Generate response using the specified or active model"""
        model = self.get_model(model_id) if model_id else self.get_active_model()
        if not model:
            return {"error": "No model available", "status": "error"}
        
        try:
            # Check cache first
            cached_response = self.cache_manager.get(query, model.id, system_message)
            if cached_response:
                logger.info(f"Cache hit for model {model.id}")
                return {
                    "response": cached_response['response'],
                    "model_id": cached_response['model_id'],
                    "cached": True,
                    "cached_at": cached_response['cached_at'],
                    "hit_count": cached_response.get('hit_count', 1),
                    "metadata": cached_response.get('metadata', {}),
                    "status": "success"
                }
            
            # Check if API key is available
            api_key = os.getenv(model.api_key_env) if model.api_key_env else None
            if model.provider != AIProvider.OLLAMA and not api_key:
                return {"error": f"API key not configured for {model.provider.value}", "status": "error"}
            
            # Route to appropriate handler
            if model.provider == AIProvider.OPENAI:
                result = await self._handle_openai(model, query, system_message, api_key)
            elif model.provider == AIProvider.ANTHROPIC:
                result = await self._handle_anthropic(model, query, system_message, api_key)
            elif model.provider == AIProvider.GOOGLE:
                result = await self._handle_google(model, query, system_message, api_key)
            elif model.provider == AIProvider.XAI:
                result = await self._handle_xai(model, query, system_message, api_key)
            elif model.provider == AIProvider.PERPLEXITY:
                result = await self._handle_perplexity(model, query, system_message, api_key)
            elif model.provider == AIProvider.OLLAMA:
                result = await self._handle_ollama(model, query, system_message)
            elif model.provider == AIProvider.COHERE:
                result = await self._handle_cohere(model, query, system_message, api_key)
            elif model.provider == AIProvider.MISTRAL:
                result = await self._handle_mistral(model, query, system_message, api_key)
            elif model.provider == AIProvider.CUSTOM:
                result = await self._handle_custom(model, query, system_message, api_key)
            else:
                return {"error": f"Unsupported provider: {model.provider.value}", "status": "error"}
            
            # Cache the response if successful
            if result.get("status") == "success" and "response" in result:
                metadata = {
                    "user_id": user_id,
                    "provider": model.provider.value,
                    "model_name": model.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "cost_per_1k_tokens": model.cost_per_1k_tokens
                }
                self.cache_manager.set(
                    query=query,
                    model_id=model.id,
                    response=result["response"],
                    system_message=system_message,
                    metadata=metadata
                )
                logger.info(f"Cached response for model {model.id}")
                result["cached"] = False
            
            return result
                
        except Exception as e:
            logger.error(f"Error generating response with model {model.id}: {e}")
            return {"error": str(e), "status": "error"}
    
    async def _handle_openai(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle OpenAI API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_anthropic(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle Anthropic API requests"""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model.model_name,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "messages": [{"role": "user", "content": query}]
        }
        
        if system_message:
            payload["system"] = system_message
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["content"][0]["text"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_google(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle Google Gemini API requests"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Google Gemini uses URL parameter for API key
        url = f"{model.endpoint}?key={api_key}"
        
        contents = []
        if system_message:
            contents.append({"role": "user", "parts": [{"text": system_message}]})
        contents.append({"role": "user", "parts": [{"text": query}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": model.temperature,
                "topP": model.top_p,
                "maxOutputTokens": model.max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["candidates"][0]["content"]["parts"][0]["text"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usageMetadata", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_xai(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle xAI API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_perplexity(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle Perplexity API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {}),
                        "citations": data.get("citations", [])
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_ollama(self, model: AIModel, query: str, system_message: str) -> Dict[str, Any]:
        """Handle Ollama (local) API requests"""
        headers = {
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": model.temperature,
                "top_p": model.top_p
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model.endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "response": data["message"]["content"],
                            "status": "success",
                            "model": model.id,
                            "usage": {
                                "total_duration": data.get("total_duration", 0),
                                "load_duration": data.get("load_duration", 0),
                                "prompt_eval_count": data.get("prompt_eval_count", 0),
                                "eval_count": data.get("eval_count", 0)
                            }
                        }
                    else:
                        error_data = await response.json()
                        return {"error": error_data.get("error", "Unknown error"), "status": "error"}
        except aiohttp.ClientConnectorError:
            return {"error": "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434", "status": "error"}
    
    async def _handle_cohere(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle Cohere API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_name,
            "message": query,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "p": model.top_p
        }
        
        if system_message:
            payload["preamble"] = system_message
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["text"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("meta", {}).get("billed_units", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("message", "Unknown error"), "status": "error"}
    
    async def _handle_mistral(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle Mistral API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {})
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_custom(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle custom endpoint requests"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if provided
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Add custom headers
        headers.update(model.custom_headers)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Try to parse OpenAI-compatible response format
                    if "choices" in data:
                        return {
                            "response": data["choices"][0]["message"]["content"],
                            "status": "success",
                            "model": model.id,
                            "usage": data.get("usage", {})
                        }
                    else:
                        return {
                            "response": str(data),
                            "status": "success",
                            "model": model.id,
                            "usage": {}
                        }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        provider_counts = {}
        for model in self.models.values():
            provider = model.provider.value
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        return {
            "total_models": len(self.models),
            "active_model": self.active_model_id,
            "providers": provider_counts,
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider.value,
                    "model_name": model.model_name,
                    "cost_per_1k_tokens": model.cost_per_1k_tokens,
                    "context_window": model.context_window,
                    "is_active": model.id == self.active_model_id
                }
                for model in self.models.values()
            ]
        }