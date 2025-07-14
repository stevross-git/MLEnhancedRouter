#!/usr/bin/env python3
"""
Configuration for ML Router and CSP Network Integration
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class NetworkIntegrationConfig:
    """Configuration for network integration"""
    
    # CSP Network Configuration
    csp_network_enabled: bool = True
    csp_network_port: int = 30405
    csp_network_node_name: str = "ml-router-bridge"
    csp_network_node_type: str = "ai_service"
    csp_network_data_dir: str = "./ml_router_network_data"
    
    # Genesis and Bootstrap Configuration
    genesis_host: str = "genesis.peoplesainetwork.com"
    genesis_port: int = 30300
    bootstrap_nodes: List[str] = field(default_factory=lambda: [
        "/ip4/genesis.peoplesainetwork.com/tcp/30300",
        "/ip4/127.0.0.1/tcp/30301",
        "/ip4/127.0.0.1/tcp/30302",
        "/ip4/127.0.0.1/tcp/30303"
    ])
    dns_seed_domain: str = "peoplesainetwork.com"
    
    # ML Router Configuration
    ml_router_enabled: bool = True
    ml_router_port: int = 5000
    ml_router_host: str = "0.0.0.0"
    ml_router_debug: bool = False
    
    # Database Configuration
    database_url: str = "sqlite:///ml_router_network.db"
    
    # Session Configuration
    session_secret: str = "ml-router-network-secret-change-in-production"
    
    # Feature Flags
    enable_dht: bool = True
    enable_mesh: bool = True
    enable_dns: bool = True
    enable_adaptive_routing: bool = True
    enable_ml_prediction: bool = True
    enable_real_time_learning: bool = True
    enable_metrics_collection: bool = True
    
    # Network Capabilities
    enable_ai_capability: bool = True
    enable_compute_capability: bool = True
    enable_relay_capability: bool = True
    enable_storage_capability: bool = False
    enable_quantum_capability: bool = False
    
    # Security Configuration
    enable_encryption: bool = True
    enable_authentication: bool = True
    key_size: int = 2048
    
    # Connection Configuration
    max_connections: int = 50
    connection_timeout: int = 30
    enable_mdns: bool = True
    enable_upnp: bool = True
    enable_nat_traversal: bool = True
    
    # Mesh Configuration
    enable_super_peers: bool = True
    max_peers: int = 50
    topology_type: str = "dynamic_partial"
    enable_multi_hop: bool = True
    max_hop_count: int = 10
    
    # ML Configuration
    training_data_size: int = 1000
    retrain_interval_hours: int = 24
    prediction_horizon_minutes: int = 30
    min_samples_for_training: int = 100
    model_accuracy_threshold: float = 0.8
    enable_ensemble_models: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "ml_router_network.log"
    enable_structured_logging: bool = True
    
    # Performance Configuration
    thread_pool_size: int = 10
    enable_async_processing: bool = True
    query_timeout_seconds: int = 60
    
    # Monitoring Configuration
    enable_prometheus_metrics: bool = False
    prometheus_port: int = 9090
    metrics_update_interval: int = 30
    
    # AI Provider Configuration (from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    
    # Redis Configuration (optional)
    redis_url: Optional[str] = None
    redis_enabled: bool = False
    
    @classmethod
    def from_env(cls) -> 'NetworkIntegrationConfig':
        """Create configuration from environment variables"""
        
        config = cls()
        
        # CSP Network Configuration
        config.csp_network_enabled = os.getenv('CSP_NETWORK_ENABLED', 'true').lower() == 'true'
        config.csp_network_port = int(os.getenv('CSP_NETWORK_PORT', '30405'))
        config.csp_network_node_name = os.getenv('CSP_NETWORK_NODE_NAME', 'ml-router-bridge')
        config.csp_network_node_type = os.getenv('CSP_NETWORK_NODE_TYPE', 'ai_service')
        config.csp_network_data_dir = os.getenv('CSP_NETWORK_DATA_DIR', './ml_router_network_data')
        
        # Genesis Configuration
        config.genesis_host = os.getenv('CSP_NETWORK_GENESIS_HOST', 'genesis.peoplesainetwork.com')
        config.genesis_port = int(os.getenv('CSP_NETWORK_GENESIS_PORT', '30300'))
        
        # ML Router Configuration
        config.ml_router_enabled = os.getenv('ML_ROUTER_ENABLED', 'true').lower() == 'true'
        config.ml_router_port = int(os.getenv('ML_ROUTER_PORT', '5000'))
        config.ml_router_host = os.getenv('ML_ROUTER_HOST', '0.0.0.0')
        config.ml_router_debug = os.getenv('ML_ROUTER_DEBUG', 'false').lower() == 'true'
        
        # Database Configuration
        config.database_url = os.getenv('DATABASE_URL', 'sqlite:///ml_router_network.db')
        
        # Session Configuration
        config.session_secret = os.getenv('SESSION_SECRET', 'ml-router-network-secret-change-in-production')
        
        # Feature Flags
        config.enable_dht = os.getenv('ENABLE_DHT', 'true').lower() == 'true'
        config.enable_mesh = os.getenv('ENABLE_MESH', 'true').lower() == 'true'
        config.enable_dns = os.getenv('ENABLE_DNS', 'true').lower() == 'true'
        config.enable_adaptive_routing = os.getenv('ENABLE_ADAPTIVE_ROUTING', 'true').lower() == 'true'
        config.enable_ml_prediction = os.getenv('ENABLE_ML_PREDICTION', 'true').lower() == 'true'
        
        # Logging Configuration
        config.log_level = os.getenv('LOG_LEVEL', 'INFO')
        config.log_file = os.getenv('LOG_FILE', 'ml_router_network.log')
        
        # AI Provider Configuration
        config.openai_api_key = os.getenv('OPENAI_API_KEY')
        config.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        config.gemini_api_key = os.getenv('GEMINI_API_KEY')
        config.xai_api_key = os.getenv('XAI_API_KEY')
        config.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Redis Configuration
        config.redis_url = os.getenv('REDIS_URL')
        config.redis_enabled = config.redis_url is not None
        
        # Connection Configuration
        config.max_connections = int(os.getenv('MAX_CONNECTIONS', '50'))
        config.connection_timeout = int(os.getenv('CONNECTION_TIMEOUT', '30'))
        
        # Performance Configuration
        config.thread_pool_size = int(os.getenv('THREAD_POOL_SIZE', '10'))
        config.query_timeout_seconds = int(os.getenv('QUERY_TIMEOUT_SECONDS', '60'))
        
        # Bootstrap nodes from environment (comma-separated)
        bootstrap_env = os.getenv('BOOTSTRAP_NODES')
        if bootstrap_env:
            config.bootstrap_nodes = [node.strip() for node in bootstrap_env.split(',')]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'csp_network': {
                'enabled': self.csp_network_enabled,
                'port': self.csp_network_port,
                'node_name': self.csp_network_node_name,
                'node_type': self.csp_network_node_type,
                'data_dir': self.csp_network_data_dir,
                'genesis_host': self.genesis_host,
                'genesis_port': self.genesis_port,
                'bootstrap_nodes': self.bootstrap_nodes,
                'dns_seed_domain': self.dns_seed_domain
            },
            'ml_router': {
                'enabled': self.ml_router_enabled,
                'port': self.ml_router_port,
                'host': self.ml_router_host,
                'debug': self.ml_router_debug
            },
            'database': {
                'url': self.database_url
            },
            'security': {
                'session_secret': self.session_secret,
                'enable_encryption': self.enable_encryption,
                'enable_authentication': self.enable_authentication,
                'key_size': self.key_size
            },
            'features': {
                'enable_dht': self.enable_dht,
                'enable_mesh': self.enable_mesh,
                'enable_dns': self.enable_dns,
                'enable_adaptive_routing': self.enable_adaptive_routing,
                'enable_ml_prediction': self.enable_ml_prediction,
                'enable_real_time_learning': self.enable_real_time_learning,
                'enable_metrics_collection': self.enable_metrics_collection
            },
            'capabilities': {
                'ai': self.enable_ai_capability,
                'compute': self.enable_compute_capability,
                'relay': self.enable_relay_capability,
                'storage': self.enable_storage_capability,
                'quantum': self.enable_quantum_capability
            },
            'connection': {
                'max_connections': self.max_connections,
                'connection_timeout': self.connection_timeout,
                'enable_mdns': self.enable_mdns,
                'enable_upnp': self.enable_upnp,
                'enable_nat_traversal': self.enable_nat_traversal
            },
            'mesh': {
                'enable_super_peers': self.enable_super_peers,
                'max_peers': self.max_peers,
                'topology_type': self.topology_type,
                'enable_multi_hop': self.enable_multi_hop,
                'max_hop_count': self.max_hop_count
            },
            'ml': {
                'training_data_size': self.training_data_size,
                'retrain_interval_hours': self.retrain_interval_hours,
                'prediction_horizon_minutes': self.prediction_horizon_minutes,
                'min_samples_for_training': self.min_samples_for_training,
                'model_accuracy_threshold': self.model_accuracy_threshold,
                'enable_ensemble_models': self.enable_ensemble_models
            },
            'logging': {
                'log_level': self.log_level,
                'log_file': self.log_file,
                'enable_structured_logging': self.enable_structured_logging
            },
            'performance': {
                'thread_pool_size': self.thread_pool_size,
                'enable_async_processing': self.enable_async_processing,
                'query_timeout_seconds': self.query_timeout_seconds
            },
            'monitoring': {
                'enable_prometheus_metrics': self.enable_prometheus_metrics,
                'prometheus_port': self.prometheus_port,
                'metrics_update_interval': self.metrics_update_interval
            },
            'ai_providers': {
                'openai_configured': self.openai_api_key is not None,
                'anthropic_configured': self.anthropic_api_key is not None,
                'gemini_configured': self.gemini_api_key is not None,
                'xai_configured': self.xai_api_key is not None,
                'perplexity_configured': self.perplexity_api_key is not None
            },
            'redis': {
                'enabled': self.redis_enabled,
                'configured': self.redis_url is not None
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        # Remove sensitive information before saving
        if 'security' in config_dict:
            config_dict['security']['session_secret'] = '[REDACTED]'
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate ports
        if not (1024 <= self.csp_network_port <= 65535):
            issues.append(f"CSP Network port {self.csp_network_port} is not in valid range (1024-65535)")
        
        if not (1024 <= self.ml_router_port <= 65535):
            issues.append(f"ML Router port {self.ml_router_port} is not in valid range (1024-65535)")
        
        if self.csp_network_port == self.ml_router_port:
            issues.append("CSP Network port and ML Router port cannot be the same")
        
        # Validate data directory
        try:
            data_dir = Path(self.csp_network_data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create data directory {self.csp_network_data_dir}: {e}")
        
        # Validate genesis configuration
        if not self.genesis_host:
            issues.append("Genesis host cannot be empty")
        
        if not (1 <= self.genesis_port <= 65535):
            issues.append(f"Genesis port {self.genesis_port} is not valid")
        
        # Validate ML configuration
        if self.training_data_size < 100:
            issues.append("Training data size should be at least 100")
        
        if self.retrain_interval_hours < 1:
            issues.append("Retrain interval should be at least 1 hour")
        
        # Validate connection configuration
        if self.max_connections < 1:
            issues.append("Max connections should be at least 1")
        
        if self.connection_timeout < 5:
            issues.append("Connection timeout should be at least 5 seconds")
        
        # Validate performance configuration
        if self.thread_pool_size < 1:
            issues.append("Thread pool size should be at least 1")
        
        if self.query_timeout_seconds < 1:
            issues.append("Query timeout should be at least 1 second")
        
        return issues

def load_config_from_file(filepath: str) -> NetworkIntegrationConfig:
    """Load configuration from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Create config from environment first, then override with file values
    config = NetworkIntegrationConfig.from_env()
    
    # Override with file values (this is a simplified version)
    # In a real implementation, you'd want more sophisticated merging
    if 'csp_network' in config_dict:
        net_config = config_dict['csp_network']
        config.csp_network_enabled = net_config.get('enabled', config.csp_network_enabled)
        config.csp_network_port = net_config.get('port', config.csp_network_port)
        config.csp_network_node_name = net_config.get('node_name', config.csp_network_node_name)
        config.genesis_host = net_config.get('genesis_host', config.genesis_host)
        config.genesis_port = net_config.get('genesis_port', config.genesis_port)
    
    if 'ml_router' in config_dict:
        ml_config = config_dict['ml_router']
        config.ml_router_enabled = ml_config.get('enabled', config.ml_router_enabled)
        config.ml_router_port = ml_config.get('port', config.ml_router_port)
        config.ml_router_debug = ml_config.get('debug', config.ml_router_debug)
    
    return config

def create_example_config():
    """Create an example configuration file"""
    config = NetworkIntegrationConfig()
    config.save_to_file('example_integration_config.json')
    print("Example configuration saved to: example_integration_config.json")

if __name__ == "__main__":
    # Create example configuration
    create_example_config()
    
    # Load configuration from environment
    config = NetworkIntegrationConfig.from_env()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  CSP Network: {'Enabled' if config.csp_network_enabled else 'Disabled'} (Port: {config.csp_network_port})")
    print(f"  ML Router: {'Enabled' if config.ml_router_enabled else 'Disabled'} (Port: {config.ml_router_port})")
    print(f"  Genesis: {config.genesis_host}:{config.genesis_port}")
    print(f"  Data Directory: {config.csp_network_data_dir}")
    print(f"  Database: {config.database_url}")
    print(f"  Redis: {'Enabled' if config.redis_enabled else 'Disabled'}")
    print(f"  AI Providers: {sum([config.openai_api_key is not None, config.anthropic_api_key is not None, config.gemini_api_key is not None])} configured")