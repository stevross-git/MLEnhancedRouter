"""
External LLM API Integration for Complex Queries
Provides specialized handling for complex queries using external LLM APIs
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import time

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ExternalProvider(Enum):
    """External LLM providers for complex queries"""
    CLAUDE_3_OPUS = "claude-3-opus"
    GPT_4_TURBO = "gpt-4-turbo"
    GEMINI_PRO = "gemini-pro"
    COMMAND_R_PLUS = "command-r-plus"
    MIXTRAL_8X7B = "mixtral-8x7b"
    LLAMA_3_70B = "llama-3-70b"

@dataclass
class ComplexQuery:
    """Represents a complex query requiring external LLM processing"""
    query: str
    complexity: QueryComplexity
    domain: str
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_analysis: bool = False
    requires_multi_step: bool = False
    context_length: int = 0
    specialized_knowledge: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.specialized_knowledge is None:
            self.specialized_knowledge = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ExternalLLMConfig:
    """Configuration for external LLM providers"""
    provider: ExternalProvider
    api_key_env: str
    endpoint: str
    model_name: str
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    supports_system_message: bool = True
    supports_streaming: bool = True
    cost_per_1k_tokens: float = 0.0
    rate_limit_rpm: int = 60
    specializations: List[str] = None
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []

class ComplexQueryAnalyzer:
    """Analyzes queries to determine complexity and routing needs"""
    
    def __init__(self):
        self.complexity_keywords = {
            QueryComplexity.SIMPLE: [
                "what is", "define", "explain briefly", "simple question"
            ],
            QueryComplexity.MODERATE: [
                "compare", "analyze", "explain how", "what are the differences"
            ],
            QueryComplexity.COMPLEX: [
                "evaluate", "synthesize", "develop a strategy", "multi-step analysis",
                "comprehensive review", "detailed comparison"
            ],
            QueryComplexity.VERY_COMPLEX: [
                "research and analyze", "create a complete framework", 
                "develop a comprehensive solution", "multi-domain analysis",
                "complex reasoning", "interdisciplinary approach"
            ]
        }
        
        self.domain_keywords = {
            "scientific": ["research", "hypothesis", "methodology", "peer review"],
            "technical": ["architecture", "implementation", "system design", "algorithm"],
            "creative": ["brainstorm", "innovative", "creative solution", "artistic"],
            "analytical": ["data analysis", "statistical", "metrics", "performance"],
            "strategic": ["business strategy", "planning", "roadmap", "vision"],
            "philosophical": ["ethics", "moral", "philosophical", "conceptual"]
        }
    
    def analyze_query(self, query: str) -> ComplexQuery:
        """Analyze a query to determine its complexity and characteristics"""
        query_lower = query.lower()
        
        # Determine complexity
        complexity = self._determine_complexity(query_lower)
        
        # Determine domain
        domain = self._determine_domain(query_lower)
        
        # Analyze requirements
        requires_reasoning = self._requires_reasoning(query_lower)
        requires_creativity = self._requires_creativity(query_lower)
        requires_analysis = self._requires_analysis(query_lower)
        requires_multi_step = self._requires_multi_step(query_lower)
        
        # Estimate context length
        context_length = len(query.split())
        
        # Identify specialized knowledge areas
        specialized_knowledge = self._identify_specialized_knowledge(query_lower)
        
        return ComplexQuery(
            query=query,
            complexity=complexity,
            domain=domain,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_analysis=requires_analysis,
            requires_multi_step=requires_multi_step,
            context_length=context_length,
            specialized_knowledge=specialized_knowledge
        )
    
    def _determine_complexity(self, query: str) -> QueryComplexity:
        """Determine query complexity based on keywords and structure"""
        word_count = len(query.split())
        
        # Check for complexity keywords
        for complexity, keywords in self.complexity_keywords.items():
            if any(keyword in query for keyword in keywords):
                return complexity
        
        # Fallback to word count heuristic
        if word_count > 50:
            return QueryComplexity.VERY_COMPLEX
        elif word_count > 30:
            return QueryComplexity.COMPLEX
        elif word_count > 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _determine_domain(self, query: str) -> str:
        """Determine the primary domain of the query"""
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                return domain
        return "general"
    
    def _requires_reasoning(self, query: str) -> bool:
        """Check if query requires complex reasoning"""
        reasoning_indicators = [
            "why", "because", "therefore", "consequently", "reasoning",
            "logic", "deduce", "infer", "conclude", "prove"
        ]
        return any(indicator in query for indicator in reasoning_indicators)
    
    def _requires_creativity(self, query: str) -> bool:
        """Check if query requires creative thinking"""
        creativity_indicators = [
            "creative", "innovative", "brainstorm", "generate ideas",
            "think outside", "novel approach", "unique solution"
        ]
        return any(indicator in query for indicator in creativity_indicators)
    
    def _requires_analysis(self, query: str) -> bool:
        """Check if query requires analytical thinking"""
        analysis_indicators = [
            "analyze", "examine", "investigate", "evaluate", "assess",
            "compare", "contrast", "breakdown", "dissect"
        ]
        return any(indicator in query for indicator in analysis_indicators)
    
    def _requires_multi_step(self, query: str) -> bool:
        """Check if query requires multi-step processing"""
        multi_step_indicators = [
            "step by step", "first", "then", "next", "finally",
            "process", "methodology", "approach", "framework"
        ]
        return any(indicator in query for indicator in multi_step_indicators)
    
    def _identify_specialized_knowledge(self, query: str) -> List[str]:
        """Identify specialized knowledge areas required"""
        knowledge_areas = {
            "machine_learning": ["ml", "ai", "neural network", "deep learning"],
            "data_science": ["data analysis", "statistics", "visualization"],
            "software_engineering": ["coding", "programming", "architecture"],
            "business": ["strategy", "management", "finance", "marketing"],
            "science": ["research", "experiment", "hypothesis", "theory"],
            "legal": ["law", "regulation", "compliance", "legal"]
        }
        
        identified = []
        for area, keywords in knowledge_areas.items():
            if any(keyword in query for keyword in keywords):
                identified.append(area)
        
        return identified

class ExternalLLMManager:
    """Manages external LLM integrations for complex queries"""
    
    def __init__(self, db=None):
        self.db = db
        self.analyzer = ComplexQueryAnalyzer()
        self.providers = self._initialize_providers()
        self.rate_limits = {}
        self.performance_metrics = {}
    
    def _initialize_providers(self) -> Dict[ExternalProvider, ExternalLLMConfig]:
        """Initialize external LLM provider configurations"""
        return {
            ExternalProvider.CLAUDE_3_OPUS: ExternalLLMConfig(
                provider=ExternalProvider.CLAUDE_3_OPUS,
                api_key_env="ANTHROPIC_API_KEY",
                endpoint="https://api.anthropic.com/v1/messages",
                model_name="claude-3-opus-20240229",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.075,
                rate_limit_rpm=60,
                specializations=["reasoning", "analysis", "creative_writing"]
            ),
            ExternalProvider.GPT_4_TURBO: ExternalLLMConfig(
                provider=ExternalProvider.GPT_4_TURBO,
                api_key_env="OPENAI_API_KEY",
                endpoint="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4-turbo-preview",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.06,
                rate_limit_rpm=60,
                specializations=["coding", "technical_analysis", "problem_solving"]
            ),
            ExternalProvider.GEMINI_PRO: ExternalLLMConfig(
                provider=ExternalProvider.GEMINI_PRO,
                api_key_env="GEMINI_API_KEY",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                model_name="gemini-pro",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.005,
                rate_limit_rpm=60,
                specializations=["research", "factual_analysis", "multi_modal"]
            ),
            ExternalProvider.COMMAND_R_PLUS: ExternalLLMConfig(
                provider=ExternalProvider.COMMAND_R_PLUS,
                api_key_env="COHERE_API_KEY",
                endpoint="https://api.cohere.ai/v1/chat",
                model_name="command-r-plus",
                max_tokens=8192,
                temperature=0.7,
                cost_per_1k_tokens=0.03,
                rate_limit_rpm=60,
                specializations=["business_analysis", "strategic_planning", "summarization"]
            )
        }
    
    async def process_complex_query(self, query: str, context: str = None, 
                                  preferred_provider: ExternalProvider = None) -> Dict[str, Any]:
        """Process a complex query using appropriate external LLM"""
        
        # Analyze query complexity
        complex_query = self.analyzer.analyze_query(query)
        
        # Select appropriate provider
        provider = preferred_provider or self._select_optimal_provider(complex_query)
        
        if not provider:
            raise ValueError("No suitable external LLM provider available")
        
        # Check rate limits
        if not self._check_rate_limit(provider):
            raise Exception(f"Rate limit exceeded for provider {provider.value}")
        
        # Process query
        start_time = time.time()
        
        try:
            response = await self._call_external_llm(provider, complex_query, context)
            
            # Record metrics
            processing_time = time.time() - start_time
            self._record_metrics(provider, complex_query, processing_time, True)
            
            return {
                "response": response,
                "provider": provider.value,
                "complexity": complex_query.complexity.value,
                "domain": complex_query.domain,
                "processing_time": processing_time,
                "specialized_knowledge": complex_query.specialized_knowledge,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._record_metrics(provider, complex_query, time.time() - start_time, False)
            logger.error(f"External LLM processing failed: {str(e)}")
            raise
    
    def _select_optimal_provider(self, complex_query: ComplexQuery) -> Optional[ExternalProvider]:
        """Select the most appropriate provider for the query"""
        
        # Priority rules based on query characteristics
        if complex_query.requires_reasoning and complex_query.complexity == QueryComplexity.VERY_COMPLEX:
            return ExternalProvider.CLAUDE_3_OPUS
        
        if complex_query.domain == "technical" and complex_query.requires_analysis:
            return ExternalProvider.GPT_4_TURBO
        
        if complex_query.domain == "scientific" or "research" in complex_query.specialized_knowledge:
            return ExternalProvider.GEMINI_PRO
        
        if complex_query.domain == "strategic" or "business" in complex_query.specialized_knowledge:
            return ExternalProvider.COMMAND_R_PLUS
        
        # Default to Claude Opus for very complex queries
        if complex_query.complexity == QueryComplexity.VERY_COMPLEX:
            return ExternalProvider.CLAUDE_3_OPUS
        
        # Default to GPT-4 Turbo for other complex queries
        return ExternalProvider.GPT_4_TURBO
    
    def _check_rate_limit(self, provider: ExternalProvider) -> bool:
        """Check if provider is within rate limits"""
        now = datetime.now()
        provider_key = provider.value
        
        if provider_key not in self.rate_limits:
            self.rate_limits[provider_key] = []
        
        # Clean old entries (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.rate_limits[provider_key] = [
            timestamp for timestamp in self.rate_limits[provider_key]
            if timestamp > cutoff
        ]
        
        # Check if under rate limit
        config = self.providers[provider]
        if len(self.rate_limits[provider_key]) < config.rate_limit_rpm:
            self.rate_limits[provider_key].append(now)
            return True
        
        return False
    
    async def _call_external_llm(self, provider: ExternalProvider, 
                                complex_query: ComplexQuery, context: str = None) -> str:
        """Call external LLM API"""
        config = self.providers[provider]
        api_key = os.environ.get(config.api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found for provider {provider.value}")
        
        # Prepare system message
        system_message = self._prepare_system_message(complex_query, context)
        
        # Prepare request based on provider
        if provider == ExternalProvider.CLAUDE_3_OPUS:
            return await self._call_anthropic(config, api_key, system_message, complex_query.query)
        elif provider == ExternalProvider.GPT_4_TURBO:
            return await self._call_openai(config, api_key, system_message, complex_query.query)
        elif provider == ExternalProvider.GEMINI_PRO:
            return await self._call_google(config, api_key, system_message, complex_query.query)
        elif provider == ExternalProvider.COMMAND_R_PLUS:
            return await self._call_cohere(config, api_key, system_message, complex_query.query)
        else:
            raise ValueError(f"Unsupported provider: {provider.value}")
    
    def _prepare_system_message(self, complex_query: ComplexQuery, context: str = None) -> str:
        """Prepare system message for external LLM"""
        base_message = f"""You are an expert AI assistant specialized in handling complex queries requiring {complex_query.complexity.value} level analysis in the {complex_query.domain} domain.

Query characteristics:
- Requires reasoning: {complex_query.requires_reasoning}
- Requires creativity: {complex_query.requires_creativity}
- Requires analysis: {complex_query.requires_analysis}
- Multi-step processing: {complex_query.requires_multi_step}
- Specialized knowledge: {', '.join(complex_query.specialized_knowledge)}

Please provide a comprehensive, well-structured response that addresses all aspects of the query."""
        
        if context:
            base_message += f"\n\nAdditional context:\n{context}"
        
        return base_message
    
    async def _call_anthropic(self, config: ExternalLLMConfig, api_key: str, 
                            system_message: str, query: str) -> str:
        """Call Anthropic API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system": system_message,
            "messages": [{"role": "user", "content": query}]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {error_text}")
    
    async def _call_openai(self, config: ExternalLLMConfig, api_key: str, 
                          system_message: str, query: str) -> str:
        """Call OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")
    
    async def _call_google(self, config: ExternalLLMConfig, api_key: str, 
                          system_message: str, query: str) -> str:
        """Call Google Gemini API"""
        headers = {
            "Content-Type": "application/json"
        }
        
        full_prompt = f"{system_message}\n\nUser query: {query}"
        
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p
            }
        }
        
        url = f"{config.endpoint}?key={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Google API error: {error_text}")
    
    async def _call_cohere(self, config: ExternalLLMConfig, api_key: str, 
                          system_message: str, query: str) -> str:
        """Call Cohere API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": config.model_name,
            "message": query,
            "preamble": system_message,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "p": config.top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Cohere API error: {error_text}")
    
    def _record_metrics(self, provider: ExternalProvider, complex_query: ComplexQuery, 
                       processing_time: float, success: bool):
        """Record performance metrics"""
        provider_key = provider.value
        
        if provider_key not in self.performance_metrics:
            self.performance_metrics[provider_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_response_time": 0,
                "complexity_distribution": {},
                "domain_distribution": {}
            }
        
        metrics = self.performance_metrics[provider_key]
        
        # Update counters
        metrics["total_requests"] += 1
        if success:
            metrics["successful_requests"] += 1
        
        # Update average response time
        current_avg = metrics["average_response_time"]
        total_requests = metrics["total_requests"]
        metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Update distributions
        complexity = complex_query.complexity.value
        domain = complex_query.domain
        
        if complexity not in metrics["complexity_distribution"]:
            metrics["complexity_distribution"][complexity] = 0
        metrics["complexity_distribution"][complexity] += 1
        
        if domain not in metrics["domain_distribution"]:
            metrics["domain_distribution"][domain] = 0
        metrics["domain_distribution"][domain] += 1
    
    def get_provider_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all providers"""
        return {
            "providers": self.performance_metrics,
            "rate_limits": {
                provider: len(timestamps) 
                for provider, timestamps in self.rate_limits.items()
            },
            "available_providers": list(self.providers.keys())
        }
    
    def is_complex_query(self, query: str) -> bool:
        """Check if a query should be handled by external LLM"""
        complex_query = self.analyzer.analyze_query(query)
        return complex_query.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]
    
    def get_recommended_provider(self, query: str) -> Optional[ExternalProvider]:
        """Get recommended provider for a query"""
        complex_query = self.analyzer.analyze_query(query)
        return self._select_optimal_provider(complex_query)

# Global instance
external_llm_manager = None

def get_external_llm_manager(db=None) -> ExternalLLMManager:
    """Get or create global external LLM manager instance"""
    global external_llm_manager
    if external_llm_manager is None:
        external_llm_manager = ExternalLLMManager(db)
    return external_llm_manager