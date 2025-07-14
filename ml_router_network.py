#!/usr/bin/env python3
"""
ML Enhanced Router with CSP Network Integration
"""

import asyncio
import logging
import threading
import sys
import os
import time
from pathlib import Path
from flask import Flask, jsonify, request, Response
from datetime import datetime
import json

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

# Import network components
try:
    from enhanced_csp.network.core.config import NetworkConfig, P2PConfig, MeshConfig, SecurityConfig
    from enhanced_csp.network.core.node import NetworkNode
    from enhanced_csp.network.core.types import NodeCapabilities
    from enhanced_csp.network import create_network
    NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Enhanced CSP Network not available: {e}")
    NETWORK_AVAILABLE = False

# Import network bridge
try:
    from network_bridge import NetworkBridge
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Network Bridge not available: {e}")
    BRIDGE_AVAILABLE = False

# Import original Flask app components
try:
    from app import app, db, initialize_router
    ML_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ML Router app not available: {e}")
    # Create minimal Flask app if original not available
    app = Flask(__name__)
    ML_ROUTER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_router_network.log')
    ]
)
logger = logging.getLogger(__name__)

# Global network instances
network_bridge = None
initialization_status = {
    "ml_router_ready": False,
    "network_bridge_ready": False,
    "initialization_time": None,
    "errors": []
}

def initialize_network_integration():
    """Initialize CSP Network integration in background thread"""
    global network_bridge, initialization_status
    
    initialization_status["initialization_time"] = datetime.now().isoformat()
    
    if not BRIDGE_AVAILABLE:
        error_msg = "Network Bridge not available - running ML Router only"
        logger.warning(error_msg)
        initialization_status["errors"].append(error_msg)
        return
    
    async def setup_network():
        global network_bridge, initialization_status
        
        try:
            logger.info("🌉 Starting Network Bridge initialization...")
            network_bridge = NetworkBridge()
            await network_bridge.start_bridge()
            
            initialization_status["network_bridge_ready"] = True
            logger.info("✅ Network Bridge initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to start Network Bridge: {e}"
            logger.error(f"❌ {error_msg}")
            initialization_status["errors"].append(error_msg)
            
    # Run in new event loop
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup_network())
    except Exception as e:
        error_msg = f"Error setting up network event loop: {e}"
        logger.error(f"❌ {error_msg}")
        initialization_status["errors"].append(error_msg)

# Enhanced API routes for network integration

@app.route('/api/network/query', methods=['POST'])
def network_query():
    """Submit query through CSP Network"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
            
        query = data['query']
        source_node = data.get('source_node', 'ml_router_client')
        
        if not network_bridge or not network_bridge.is_running:
            # Fallback to regular processing
            return jsonify({
                'status': 'fallback',
                'message': 'Network bridge not available, processed locally',
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'routing_method': 'local_fallback'
            }), 200
            
        # Route query through CSP Network
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                network_bridge.route_query_via_network(query, source_node)
            )
            
            return jsonify({
                'status': 'success',
                'query': query,
                'network_result': result,
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Network query error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/network/status')
def network_status():
    """Get comprehensive CSP Network status"""
    if not NETWORK_AVAILABLE:
        return jsonify({
            'status': 'not_available', 
            'reason': 'CSP Network not imported',
            'network_available': False,
            'bridge_available': BRIDGE_AVAILABLE,
            'ml_router_available': ML_ROUTER_AVAILABLE
        })
        
    if not network_bridge:
        return jsonify({
            'status': 'not_initialized',
            'network_available': True,
            'bridge_available': BRIDGE_AVAILABLE,
            'initialization_status': initialization_status
        })
        
    try:
        bridge_status = network_bridge.get_bridge_status()
        
        return jsonify({
            'status': 'running' if bridge_status['is_running'] else 'stopped',
            'bridge_status': bridge_status,
            'initialization_status': initialization_status,
            'network_available': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'error': str(e),
            'network_available': True,
            'bridge_available': BRIDGE_AVAILABLE
        })

@app.route('/api/network/peers')
def network_peers():
    """Get connected peers information"""
    if not network_bridge or not network_bridge.is_running:
        return jsonify({'peers': [], 'error': 'Network bridge not running'})
        
    try:
        bridge_status = network_bridge.get_bridge_status()
        peers_info = bridge_status.get('peers', [])
        
        return jsonify({
            'peers': peers_info,
            'peer_count': bridge_status.get('peer_count', 0),
            'node_id': bridge_status.get('node_id', 'unknown'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'peers': []})

@app.route('/api/network/metrics')
def network_metrics():
    """Get network performance metrics"""
    if not network_bridge or not network_bridge.is_running:
        return jsonify({'error': 'Network bridge not running'})
        
    try:
        metrics = network_bridge.get_network_metrics()
        bridge_status = network_bridge.get_bridge_status()
        
        combined_metrics = {
            'network_metrics': metrics,
            'routing_stats': {
                'queries_processed': bridge_status.get('queries_processed', 0),
                'network_routes_used': bridge_status.get('network_routes_used', 0),
                'fallback_routes_used': bridge_status.get('fallback_routes_used', 0),
                'uptime_seconds': bridge_status.get('uptime_seconds', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(combined_metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/network/health')
def network_health():
    """Comprehensive health check for network integration"""
    health_status = {
        'overall_status': 'unknown',
        'components': {
            'csp_network': {
                'available': NETWORK_AVAILABLE,
                'status': 'unknown'
            },
            'network_bridge': {
                'available': BRIDGE_AVAILABLE,
                'status': 'unknown'
            },
            'ml_router': {
                'available': ML_ROUTER_AVAILABLE,
                'status': 'unknown'
            }
        },
        'initialization': initialization_status,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check network bridge status
    if network_bridge:
        try:
            bridge_status = network_bridge.get_bridge_status()
            health_status['components']['network_bridge']['status'] = 'running' if bridge_status['is_running'] else 'stopped'
            health_status['components']['csp_network']['status'] = 'connected' if bridge_status['csp_network_connected'] else 'disconnected'
            health_status['components']['ml_router']['status'] = 'ready' if bridge_status['ml_router_ready'] else 'not_ready'
        except Exception as e:
            health_status['components']['network_bridge']['status'] = f'error: {e}'
    else:
        health_status['components']['network_bridge']['status'] = 'not_initialized'
    
    # Determine overall status
    if (health_status['components']['network_bridge']['status'] == 'running' and 
        health_status['components']['csp_network']['status'] == 'connected'):
        health_status['overall_status'] = 'healthy'
    elif health_status['components']['ml_router']['available']:
        health_status['overall_status'] = 'degraded'  # ML Router works but no network
    else:
        health_status['overall_status'] = 'unhealthy'
    
    return jsonify(health_status)

@app.route('/api/network/shutdown', methods=['POST'])
def shutdown_network():
    """Gracefully shutdown network components"""
    global network_bridge
    
    if not network_bridge:
        return jsonify({'status': 'no_network_to_shutdown'})
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(network_bridge.stop_bridge())
        network_bridge = None
        
        return jsonify({
            'status': 'shutdown_complete',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'shutdown_error', 'error': str(e)})

# Enhanced dashboard route
@app.route('/network-dashboard')
def network_dashboard():
    """Network integration dashboard"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Router Network Integration Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status-card {{ border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 8px; }}
            .healthy {{ border-color: #4CAF50; background-color: #f9fff9; }}
            .degraded {{ border-color: #FF9800; background-color: #fff9f0; }}
            .unhealthy {{ border-color: #f44336; background-color: #fff0f0; }}
            .code {{ background-color: #f5f5f5; padding: 10px; font-family: monospace; }}
        </style>
        <script>
            function refreshStatus() {{
                fetch('/api/network/health')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('status-content').innerHTML = 
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }});
            }}
            setInterval(refreshStatus, 5000);
            window.onload = refreshStatus;
        </script>
    </head>
    <body>
        <h1>🌐 ML Router Network Integration Dashboard</h1>
        
        <div class="status-card">
            <h2>Quick Status</h2>
            <p>Network Available: {NETWORK_AVAILABLE}</p>
            <p>Bridge Available: {BRIDGE_AVAILABLE}</p>
            <p>ML Router Available: {ML_ROUTER_AVAILABLE}</p>
            <p>Bridge Running: {network_bridge.is_running if network_bridge else False}</p>
        </div>
        
        <div class="status-card">
            <h2>Live Status (refreshes every 5 seconds)</h2>
            <div id="status-content" class="code">Loading...</div>
        </div>
        
        <div class="status-card">
            <h2>API Endpoints</h2>
            <ul>
                <li><a href="/api/network/status">Network Status</a></li>
                <li><a href="/api/network/health">Health Check</a></li>
                <li><a href="/api/network/peers">Connected Peers</a></li>
                <li><a href="/api/network/metrics">Performance Metrics</a></li>
            </ul>
        </div>
        
        <div class="status-card">
            <h2>Test Network Query</h2>
            <button onclick="testQuery()">Send Test Query</button>
            <div id="query-result" class="code" style="margin-top: 10px;"></div>
            <script>
                function testQuery() {{
                    fetch('/api/network/query', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{'query': 'Test network routing query'}})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('query-result').innerHTML = 
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }});
                }}
            </script>
        </div>
    </body>
    </html>
    """

# Add basic routes if ML Router not available
if not ML_ROUTER_AVAILABLE:
    @app.route('/')
    def index():
        return jsonify({
            'message': 'ML Router Network Integration Server',
            'status': 'running',
            'ml_router_available': False,
            'network_available': NETWORK_AVAILABLE,
            'bridge_available': BRIDGE_AVAILABLE,
            'dashboard': '/network-dashboard'
        })

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'flask': 'running',
            'network': 'available' if NETWORK_AVAILABLE else 'unavailable',
            'bridge': 'available' if BRIDGE_AVAILABLE else 'unavailable',
            'ml_router': 'available' if ML_ROUTER_AVAILABLE else 'unavailable'
        }
    })

def main():
    """Main entry point"""
    print("🚀 ML Enhanced Router with CSP Network Integration")
    print("=" * 60)
    print(f"🤖 ML Router Available: {ML_ROUTER_AVAILABLE}")
    print(f"🌐 CSP Network Available: {NETWORK_AVAILABLE}")
    print(f"🌉 Network Bridge Available: {BRIDGE_AVAILABLE}")
    print()
    
    # Initialize original ML router if available
    if ML_ROUTER_AVAILABLE:
        print("🤖 Initializing ML Enhanced Router...")
        try:
            initialize_router()
            initialization_status["ml_router_ready"] = True
            print("✅ ML Router initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize ML Router: {e}"
            print(f"❌ {error_msg}")
            initialization_status["errors"].append(error_msg)
    else:
        print("⚠️  Running without ML Router - using minimal Flask app")
    
    # Initialize network integration in background
    if NETWORK_AVAILABLE and BRIDGE_AVAILABLE:
        print("🌐 Starting CSP Network integration...")
        bridge_thread = threading.Thread(target=initialize_network_integration)
        bridge_thread.daemon = True
        bridge_thread.start()
        
        # Give the bridge time to initialize
        time.sleep(2)
    else:
        print("⚠️  Running without CSP Network integration")
    
    print()
    print("🚀 Starting server...")
    print("📍 ML Router: http://localhost:5000")
    print("🌐 Network Dashboard: http://localhost:5000/network-dashboard")
    print("🔍 Network Status: http://localhost:5000/api/network/status")
    print("❤️  Health Check: http://localhost:5000/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if network_bridge:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(network_bridge.stop_bridge())
        print("👋 Shutdown complete")

if __name__ == '__main__':
    main()