import os
from dataclasses import dataclass
from enum import Enum

class QueryCategory(Enum):
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    CODING = "coding"
    RESEARCH = "research"
    PHILOSOPHICAL = "philosophical"
    PRACTICAL = "practical"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"

@dataclass
class EnhancedRouterConfig:
    """Enhanced configuration for production deployment"""
    # Routing parameters
    min_confidence_threshold: float = 0.7
    max_agents_per_query: int = 5
    consensus_threshold: float = 0.8
    ml_fallback_threshold: float = 0.7
    
    # Performance parameters
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    max_retries: int = 3
    retry_delay_base: float = 1.0
    
    # Load balancing
    max_concurrent_queries_per_agent: int = 50
    load_penalty_factor: float = 0.2
    
    # Timeouts
    agent_timeout_seconds: float = 10.0
    
    # Rate limiting
    rate_limit_per_minute: int = 1000
    rate_limit_window_size: int = 60
    global_rate_limit_per_minute: int = 5000
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_cluster_mode: bool = False
    use_redis_cache: bool = True
    
    # ML Model configuration
    ml_model_path: str = "./models/query_classifier"
    use_ml_classification: bool = True
    semantic_similarity_model: str = "all-MiniLM-L6-v2"
    external_llm_api: str = ""
    external_llm_api_key: str = ""
    
    # Service discovery
    service_registry_url: str = "http://localhost:8500/v1/agent/services"
    discovery_interval_seconds: int = 30
    
    # Metrics
    metrics_port: int = 9090
    
    # Security
    jwt_secret: str = os.getenv("ROUTER_JWT_SECRET", "dev-secret")
    api_auth_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'EnhancedRouterConfig':
        """Load configuration from environment variables"""
        return cls(
            min_confidence_threshold=float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.7")),
            max_agents_per_query=int(os.getenv("ROUTER_MAX_AGENTS", "5")),
            consensus_threshold=float(os.getenv("ROUTER_CONSENSUS_THRESHOLD", "0.8")),
            ml_fallback_threshold=float(os.getenv("ROUTER_ML_FALLBACK_THRESHOLD", "0.7")),
            cache_ttl_seconds=int(os.getenv("ROUTER_CACHE_TTL", "3600")),
            cache_max_size=int(os.getenv("ROUTER_CACHE_SIZE", "10000")),
            max_retries=int(os.getenv("ROUTER_MAX_RETRIES", "3")),
            retry_delay_base=float(os.getenv("ROUTER_RETRY_DELAY", "1.0")),
            max_concurrent_queries_per_agent=int(os.getenv("ROUTER_MAX_CONCURRENT", "50")),
            load_penalty_factor=float(os.getenv("ROUTER_LOAD_PENALTY", "0.2")),
            agent_timeout_seconds=float(os.getenv("ROUTER_AGENT_TIMEOUT", "10.0")),
            rate_limit_per_minute=int(os.getenv("ROUTER_RATE_LIMIT", "1000")),
            global_rate_limit_per_minute=int(os.getenv("ROUTER_GLOBAL_RATE_LIMIT", "5000")),
            redis_url=os.getenv("ROUTER_REDIS_URL", "redis://localhost:6379"),
            redis_cluster_mode=os.getenv("ROUTER_REDIS_CLUSTER", "false").lower() == "true",
            use_redis_cache=os.getenv("ROUTER_USE_REDIS", "true").lower() == "true",
            ml_model_path=os.getenv("ROUTER_ML_MODEL_PATH", "./models/query_classifier"),
            use_ml_classification=os.getenv("ROUTER_USE_ML", "true").lower() == "true",
            semantic_similarity_model=os.getenv("ROUTER_SIMILARITY_MODEL", "all-MiniLM-L6-v2"),
            external_llm_api=os.getenv("ROUTER_EXTERNAL_LLM_API", ""),
            external_llm_api_key=os.getenv("ROUTER_EXTERNAL_LLM_KEY", ""),
            service_registry_url=os.getenv("ROUTER_SERVICE_REGISTRY", "http://localhost:8500/v1/agent/services"),
            discovery_interval_seconds=int(os.getenv("ROUTER_DISCOVERY_INTERVAL", "30")),
            metrics_port=int(os.getenv("ROUTER_METRICS_PORT", "9090")),
            jwt_secret=os.getenv("ROUTER_JWT_SECRET", "dev-secret"),
            api_auth_enabled=os.getenv("ROUTER_AUTH_ENABLED", "true").lower() == "true"
        )
