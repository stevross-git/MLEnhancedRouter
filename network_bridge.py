#!/usr/bin/env python3
"""
Network Bridge - Connects ML Enhanced Router to Enhanced CSP Network
"""

import asyncio
import logging
import json
import sys
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Import CSP Network components
try:
    from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from enhanced_csp.network.core.node import NetworkNode
    from enhanced_csp.network.core.types import NodeID, NodeCapabilities
    from enhanced_csp.network import create_network
    from enhanced_csp.network.ml_routing import MLRoutePredictor, MLConfig, NetworkMetrics
    CSP_NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced CSP Network not available: {e}")
    CSP_NETWORK_AVAILABLE = False

# Import ML Router components
try:
    from ml_router import MLEnhancedQueryRouter, QueryCategory
    from config import EnhancedRouterConfig
    ML_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ML Router not available: {e}")
    ML_ROUTER_AVAILABLE = False

logger = logging.getLogger(__name__)

class NetworkBridge:
    """Bridge between ML Router and CSP Network"""
    
    def __init__(self, ml_router_port: int = 5000):
        self.ml_router_port = ml_router_port
        self.network_node: Optional[NetworkNode] = None
        self.csp_network = None
        self.ml_router: Optional[MLEnhancedQueryRouter] = None
        self.ml_predictor: Optional[MLRoutePredictor] = None
        self.is_running = False
        self.start_time = time.time()
        
        # Network statistics
        self.queries_processed = 0
        self.network_routes_used = 0
        self.fallback_routes_used = 0
        
        # Node routing map
        self.node_capabilities = {
            "compute": ["technical", "coding", "mathematical"],
            "storage": ["research", "analysis", "data"],
            "relay": ["conversational", "general"],
            "ai_service": ["creative", "philosophical", "educational"]
        }
        
    async def initialize_network(self):
        """Initialize CSP Network node for ML Router"""
        if not CSP_NETWORK_AVAILABLE:
            logger.warning("CSP Network not available - bridge will run in fallback mode")
            return False
            
        logger.info("🌐 Initializing CSP Network node for ML Router...")
        
        try:
            # Create network config for ML Router node
            config = NetworkConfig()
            config.node_name = "ml-router-bridge"
            config.node_type = "ai_service"
            config.listen_port = 30405  # ML Router network port
            config.data_dir = Path("./ml_router_network_data")
            config.data_dir.mkdir(exist_ok=True)
            
            # P2P Configuration
            config.p2p = P2PConfig()
            config.p2p.listen_address = "0.0.0.0"
            config.p2p.listen_port = 30405
            config.p2p.enable_mdns = True
            config.p2p.enable_upnp = True
            config.p2p.enable_nat_traversal = True
            config.p2p.connection_timeout = 30
            config.p2p.max_connections = 50
            
            # Connect to existing network
            config.p2p.bootstrap_nodes = [
                "/ip4/genesis.peoplesainetwork.com/tcp/30300",
                "/ip4/127.0.0.1/tcp/30301",  # Local CSP network node
                "/ip4/127.0.0.1/tcp/30302",
                "/ip4/127.0.0.1/tcp/30303",
            ]
            config.p2p.dns_seed_domain = "peoplesainetwork.com"
            
            # Mesh Configuration
            config.mesh = MeshConfig()
            config.mesh.enable_super_peers = True
            config.mesh.max_peers = 50
            config.mesh.topology_type = "dynamic_partial"
            config.mesh.enable_multi_hop = True
            config.mesh.max_hop_count = 10
            
            # Security Configuration
            config.security = SecurityConfig()
            config.security.enable_encryption = True
            config.security.enable_authentication = True
            
            # Enable necessary capabilities
            if hasattr(config, 'capabilities'):
                config.capabilities = NodeCapabilities(
                    ai=True,
                    compute=True,
                    relay=True,
                    storage=False,
                    quantum=False,
                    dns=True,
                    bootstrap=False,
                    mesh_routing=True,
                    nat_traversal=True
                )
            
            # Feature flags
            config.enable_dht = True
            config.enable_mesh = True
            config.enable_dns = True
            config.enable_adaptive_routing = True
            config.enable_routing = True
            config.enable_metrics = True
            config.enable_ai = True
            
            # Initialize network
            self.csp_network = create_network(config)
            await self.csp_network.start()
            
            # Get the default node
            self.network_node = self.csp_network.get_node("default")
            
            logger.info("✅ CSP Network node started for ML Router")
            logger.info(f"🆔 Node ID: {self.network_node.node_id if self.network_node else 'Unknown'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CSP Network: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
        
    async def initialize_ml_router(self):
        """Initialize ML Enhanced Router"""
        if not ML_ROUTER_AVAILABLE:
            logger.warning("ML Router not available")
            return False
            
        logger.info("🤖 Initializing ML Enhanced Router...")
        
        try:
            router_config = EnhancedRouterConfig.from_env()
            self.ml_router = MLEnhancedQueryRouter(router_config)
            await self.ml_router.initialize()
            
            # Initialize ML predictor for network routing
            if CSP_NETWORK_AVAILABLE:
                ml_config = MLConfig(
                    training_data_size=1000,
                    retrain_interval_hours=24,
                    enable_real_time_learning=True,
                    enable_ensemble_models=True
                )
                self.ml_predictor = MLRoutePredictor(ml_config)
            
            logger.info("✅ ML Enhanced Router initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Router: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
        
    async def start_bridge(self):
        """Start the network bridge"""
        logger.info("🌉 Starting Network Bridge...")
        
        network_ok = await self.initialize_network()
        ml_router_ok = await self.initialize_ml_router()
        
        if not network_ok and not ml_router_ok:
            raise RuntimeError("Both network and ML router initialization failed")
        
        self.is_running = True
        
        if network_ok and ml_router_ok:
            logger.info("🚀 Network Bridge started successfully with full integration!")
        elif ml_router_ok:
            logger.info("🚀 Network Bridge started in ML Router only mode")
        else:
            logger.info("🚀 Network Bridge started in network only mode")
        
    async def stop_bridge(self):
        """Stop the network bridge"""
        logger.info("🛑 Stopping Network Bridge...")
        
        self.is_running = False
        
        if self.csp_network:
            try:
                await self.csp_network.stop()
            except Exception as e:
                logger.error(f"Error stopping CSP Network: {e}")
                
        logger.info("👋 Network Bridge stopped")
        
    async def route_query_via_network(self, query: str, source_node: str = None) -> Dict[str, Any]:
        """Route query through CSP Network to appropriate compute nodes"""
        if not self.is_running:
            raise RuntimeError("Bridge not running")
            
        self.queries_processed += 1
        start_time = time.time()
        
        try:
            # Use ML Router to classify query if available
            if self.ml_router:
                category, confidence = await self.ml_router.ml_classifier.classify_query(query)
                category_str = category.value if hasattr(category, 'value') else str(category)
            else:
                # Fallback classification
                category_str = self._simple_classify_query(query)
                confidence = 0.5
            
            # Determine target nodes based on category
            target_nodes = self._get_target_nodes_for_category(category_str)
            
            # Route through CSP Network if available
            if self.network_node and self.csp_network:
                result = await self._send_network_query(query, target_nodes)
                self.network_routes_used += 1
                routing_method = "csp_network"
            else:
                # Fallback to local processing
                result = await self._process_query_locally(query, category_str)
                self.fallback_routes_used += 1
                routing_method = "local_fallback"
            
            processing_time = time.time() - start_time
            
            # Record metrics if ML predictor is available
            if self.ml_predictor:
                metric = NetworkMetrics(
                    timestamp=time.time(),
                    source="ml_router_bridge",
                    destination=",".join(target_nodes),
                    latency_ms=processing_time * 1000,
                    bandwidth_mbps=1.0,  # Placeholder
                    packet_loss=0.0,
                    jitter_ms=0.0,
                    route_hops=len(target_nodes),
                    congestion_score=0.1,
                    time_of_day=datetime.now().hour,
                    day_of_week=datetime.now().weekday(),
                    network_load=0.5
                )
                self.ml_predictor.data_collector.add_metric(metric)
            
            return {
                "category": category_str,
                "confidence": confidence,
                "routed_to": target_nodes,
                "result": result,
                "routing_method": routing_method,
                "processing_time_ms": processing_time * 1000,
                "node_id": str(self.network_node.node_id)[:16] if self.network_node else None
            }
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            self.fallback_routes_used += 1
            return {
                "category": "error",
                "confidence": 0.0,
                "routed_to": [],
                "result": f"Error processing query: {str(e)}",
                "routing_method": "error_fallback",
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
    def _simple_classify_query(self, query: str) -> str:
        """Simple keyword-based query classification fallback"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["code", "program", "function", "algorithm"]):
            return "coding"
        elif any(word in query_lower for word in ["calculate", "math", "equation", "formula"]):
            return "mathematical"
        elif any(word in query_lower for word in ["analyze", "data", "research", "study"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "write", "design", "imagine"]):
            return "creative"
        elif any(word in query_lower for word in ["technical", "system", "architecture"]):
            return "technical"
        else:
            return "conversational"
            
    def _get_target_nodes_for_category(self, category: str) -> List[str]:
        """Get target nodes based on query category"""
        for node_type, categories in self.node_capabilities.items():
            if category in categories:
                return [f"{node_type}_node"]
        
        # Default to relay nodes
        return ["relay_node"]
        
    async def _send_network_query(self, query: str, target_nodes: List[str]) -> str:
        """Send query through CSP Network"""
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Find actual nodes of the specified types
        # 2. Send messages through the network protocol
        # 3. Handle responses and aggregation
        
        if self.network_node and hasattr(self.network_node, 'peers'):
            peer_count = len(self.network_node.peers)
            return f"Query processed via CSP Network. Connected to {peer_count} peers. Target nodes: {', '.join(target_nodes)}"
        else:
            return f"Query processed via CSP Network (standalone). Target nodes: {', '.join(target_nodes)}"
            
    async def _process_query_locally(self, query: str, category: str) -> str:
        """Process query locally without network"""
        return f"Query processed locally. Category: {category}. Query: {query[:100]}..."
        
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive bridge status"""
        uptime = time.time() - self.start_time
        
        status = {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "csp_network_available": CSP_NETWORK_AVAILABLE,
            "ml_router_available": ML_ROUTER_AVAILABLE,
            "csp_network_connected": self.csp_network is not None,
            "ml_router_ready": self.ml_router is not None,
            "queries_processed": self.queries_processed,
            "network_routes_used": self.network_routes_used,
            "fallback_routes_used": self.fallback_routes_used
        }
        
        if self.network_node:
            status["node_id"] = str(self.network_node.node_id)[:16]
            if hasattr(self.network_node, 'peers'):
                status["peer_count"] = len(self.network_node.peers)
                status["peers"] = [str(peer)[:16] for peer in self.network_node.peers]
            
        return status
        
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics"""
        if not self.ml_predictor:
            return {"error": "ML predictor not available"}
            
        metrics = {
            "total_samples": len(self.ml_predictor.data_collector.metrics_history),
            "routes_tracked": len(self.ml_predictor.data_collector.route_performance),
            "prediction_cache_size": len(self.ml_predictor.prediction_cache),
            "model_trained": hasattr(self.ml_predictor, 'models') and len(self.ml_predictor.models) > 0
        }
        
        if self.ml_predictor.data_collector.metrics_history:
            recent_metric = self.ml_predictor.data_collector.metrics_history[-1]
            metrics["latest_latency_ms"] = recent_metric.latency_ms
            metrics["latest_bandwidth_mbps"] = recent_metric.bandwidth_mbps
            
        return metrics