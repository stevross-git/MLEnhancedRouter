// Dashboard JavaScript
let categoryChart = null;
let cacheChart = null;
let externalLLMChart = null;
let collaborativeChart = null;
let responseTimeChart = null;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    // Update every 10 seconds
    setInterval(updateDashboard, 10000);
});

async function initializeDashboard() {
    await updateDashboard();
    initializeCharts();
}

async function updateDashboard() {
    try {
        // Load all metrics
        const [
            statsResponse, 
            healthResponse, 
            cacheResponse,
            ragResponse,
            collaborativeResponse,
            externalLLMResponse,
            externalLLMProvidersResponse,
            externalLLMMetricsResponse
        ] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/health'),
            fetch('/api/cache/stats'),
            fetch('/api/rag/stats'),
            fetch('/api/collaborate/sessions'),
            fetch('/api/external-llm/providers'),
            fetch('/api/external-llm/providers'),
            fetch('/api/external-llm/metrics')
        ]);
        
        const stats = await statsResponse.json();
        const health = await healthResponse.json();
        const cacheStats = await cacheResponse.json();
        const ragStats = await ragResponse.json();
        const collaborativeStats = await collaborativeResponse.json();
        const externalLLMProviders = await externalLLMProvidersResponse.json();
        const externalLLMMetrics = await externalLLMMetricsResponse.json();
        
        // Update all metrics
        updateMetricsCards(stats, cacheStats, ragStats, collaborativeStats, externalLLMMetrics);
        updateSystemHealth(health);
        updateCharts(stats, cacheStats, externalLLMMetrics, collaborativeStats);
        updateExternalLLMProviders(externalLLMProviders);
        updateCollaborativeAgents(collaborativeStats);
        updateRecentActivity();
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showError('Failed to load dashboard data');
    }
}

function updateMetricsCards(stats, cacheStats, ragStats, collaborativeStats, externalLLMMetrics) {
    // Main metrics
    document.getElementById('totalQueries').textContent = stats.total_queries || 0;
    
    const successRate = stats.total_queries > 0 ? 
        ((stats.successful_routes / stats.total_queries) * 100).toFixed(1) : 0;
    document.getElementById('successRate').textContent = successRate + '%';
    
    document.getElementById('activeAgents').textContent = stats.active_agents || 0;
    
    const avgResponse = stats.avg_response_time ? 
        (stats.avg_response_time * 1000).toFixed(0) : 0;
    document.getElementById('avgResponse').textContent = avgResponse + 'ms';
    
    // Collaborative sessions
    document.getElementById('collaborativeSessions').textContent = 
        collaborativeStats.active_sessions || 0;
    
    // External LLM calls
    document.getElementById('externalLLMCalls').textContent = 
        externalLLMMetrics.total_queries || 0;
    
    // Cache hit rate
    const cacheHitRate = cacheStats.total_requests > 0 ? 
        ((cacheStats.cache_hits / cacheStats.total_requests) * 100).toFixed(1) : 0;
    document.getElementById('cacheHitRate').textContent = cacheHitRate + '%';
    
    // RAG documents
    document.getElementById('ragDocuments').textContent = ragStats.total_documents || 0;
    
    // Memory usage (approximate)
    const memoryUsage = Math.round((cacheStats.memory_usage || 0) / 1024 / 1024);
    document.getElementById('memoryUsage').textContent = memoryUsage + ' MB';
    
    // API costs (calculated from external LLM usage)
    const apiCosts = calculateAPICosts(externalLLMMetrics);
    document.getElementById('apiCosts').textContent = '$' + apiCosts.toFixed(2);
}

function calculateAPICosts(metrics) {
    // Simple cost calculation based on token usage
    // This would be more sophisticated in a real implementation
    return (metrics.total_queries || 0) * 0.01; // $0.01 per query estimate
}

function updateExternalLLMProviders(providers) {
    const container = document.getElementById('externalLLMProviders');
    if (!container) return;
    
    container.innerHTML = '';
    
    providers.providers.forEach(provider => {
        const providerCard = document.createElement('div');
        providerCard.className = 'col-md-4 mb-3';
        
        const statusBadge = provider.api_key_available ? 
            '<span class="badge bg-success">Active</span>' : 
            '<span class="badge bg-warning">No API Key</span>';
        
        providerCard.innerHTML = `
            <div class="card h-100">
                <div class="card-body">
                    <h6 class="card-title">${provider.name}</h6>
                    <p class="card-text small">
                        <strong>Max Tokens:</strong> ${provider.max_tokens.toLocaleString()}<br>
                        <strong>Cost:</strong> $${provider.cost_per_1k_tokens}/1k tokens<br>
                        <strong>Rate Limit:</strong> ${provider.rate_limit_rpm} RPM
                    </p>
                    ${statusBadge}
                </div>
            </div>
        `;
        
        container.appendChild(providerCard);
    });
}

function updateCollaborativeAgents(stats) {
    const container = document.getElementById('collaborativeAgents');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Mock collaborative agents data - in real implementation, this would come from API
    const agents = [
        { name: 'Analyst', status: 'active', sessions: 2, model: 'GPT-4' },
        { name: 'Creative', status: 'active', sessions: 1, model: 'Claude-3' },
        { name: 'Technical', status: 'idle', sessions: 0, model: 'GPT-4' },
        { name: 'Researcher', status: 'active', sessions: 3, model: 'Gemini' },
        { name: 'Synthesizer', status: 'active', sessions: 1, model: 'Claude-3' }
    ];
    
    agents.forEach(agent => {
        const agentCard = document.createElement('div');
        agentCard.className = 'col-md-2 mb-3';
        
        const statusBadge = agent.status === 'active' ? 
            '<span class="badge bg-success">Active</span>' : 
            '<span class="badge bg-secondary">Idle</span>';
        
        agentCard.innerHTML = `
            <div class="card h-100">
                <div class="card-body">
                    <h6 class="card-title">${agent.name}</h6>
                    <p class="card-text small">
                        <strong>Sessions:</strong> ${agent.sessions}<br>
                        <strong>Model:</strong> ${agent.model}
                    </p>
                    ${statusBadge}
                </div>
            </div>
        `;
        
        container.appendChild(agentCard);
    });
}

function updateRecentActivity() {
    const tbody = document.getElementById('recentActivity');
    if (!tbody) return;
    
    // Mock recent activity data
    const activities = [
        { time: '2 min ago', type: 'Query', description: 'Complex analysis query processed', status: 'success', duration: '1.2s' },
        { time: '5 min ago', type: 'Collaborative', description: 'Multi-agent session completed', status: 'success', duration: '3.8s' },
        { time: '8 min ago', type: 'External LLM', description: 'Claude-3 Opus query processed', status: 'success', duration: '2.1s' },
        { time: '12 min ago', type: 'RAG', description: 'Document search and retrieval', status: 'success', duration: '0.9s' },
        { time: '15 min ago', type: 'Cache', description: 'Cache hit for similar query', status: 'success', duration: '0.1s' }
    ];
    
    tbody.innerHTML = '';
    
    activities.forEach(activity => {
        const row = document.createElement('tr');
        
        const statusClass = activity.status === 'success' ? 'success' : 'danger';
        const statusIcon = activity.status === 'success' ? 'check-circle' : 'exclamation-triangle';
        
        row.innerHTML = `
            <td><small class="text-muted">${activity.time}</small></td>
            <td><span class="badge bg-info">${activity.type}</span></td>
            <td>${activity.description}</td>
            <td><i class="fas fa-${statusIcon} text-${statusClass}"></i></td>
            <td><small class="text-muted">${activity.duration}</small></td>
        `;
        
        tbody.appendChild(row);
    });
}

function updateSystemHealth(health) {
    const routerStatus = document.getElementById('routerStatus');
    const mlStatus = document.getElementById('mlStatus');
    
    if (health.status === 'healthy') {
        routerStatus.innerHTML = '<span class="badge bg-success">Online</span>';
    } else {
        routerStatus.innerHTML = '<span class="badge bg-danger">Offline</span>';
    }
    
    if (health.ml_classifier_initialized) {
        mlStatus.innerHTML = '<span class="badge bg-success">Initialized</span>';
    } else {
        mlStatus.innerHTML = '<span class="badge bg-warning">Not Initialized</span>';
    }
}

function initializeCharts() {
    // Category distribution chart
    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    categoryChart = new Chart(categoryCtx, {
        type: 'doughnut',
        data: {
            labels: ['Analysis', 'Creative', 'Technical', 'Research', 'Other'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#007bff',
                    '#28a745',
                    '#ffc107',
                    '#dc3545',
                    '#6c757d'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // External LLM distribution chart
    const externalLLMCtx = document.getElementById('externalLLMChart').getContext('2d');
    externalLLMChart = new Chart(externalLLMCtx, {
        type: 'pie',
        data: {
            labels: ['Claude-3', 'GPT-4', 'Gemini', 'Cohere', 'Other'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#ff6384',
                    '#36a2eb',
                    '#ffcd56',
                    '#4bc0c0',
                    '#9966ff'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    // Collaborative agents chart
    const collaborativeCtx = document.getElementById('collaborativeChart').getContext('2d');
    collaborativeChart = new Chart(collaborativeCtx, {
        type: 'bar',
        data: {
            labels: ['Analyst', 'Creative', 'Technical', 'Researcher', 'Synthesizer'],
            datasets: [{
                label: 'Active Sessions',
                data: [0, 0, 0, 0, 0],
                backgroundColor: '#007bff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Response time trends chart
    const responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
    responseTimeChart = new Chart(responseTimeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Response Time (ms)',
                data: [],
                borderColor: '#007bff',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Cache performance chart
    const cacheCtx = document.getElementById('cacheChart').getContext('2d');
    cacheChart = new Chart(cacheCtx, {
        type: 'bar',
        data: {
            labels: ['Cache Hits', 'Cache Misses'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#28a745', '#dc3545']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function updateCharts(stats, cacheStats, externalLLMMetrics, collaborativeStats) {
    // Update category chart
    if (categoryChart && stats.category_distribution) {
        const categories = Object.keys(stats.category_distribution);
        const values = Object.values(stats.category_distribution);
        
        categoryChart.data.labels = categories;
        categoryChart.data.datasets[0].data = values;
        categoryChart.update();
    }
    
    // Update external LLM chart
    if (externalLLMChart && externalLLMMetrics.provider_breakdown) {
        const providers = Object.keys(externalLLMMetrics.provider_breakdown);
        const values = providers.map(provider => 
            externalLLMMetrics.provider_breakdown[provider].total_requests || 0
        );
        
        externalLLMChart.data.labels = providers;
        externalLLMChart.data.datasets[0].data = values;
        externalLLMChart.update();
    }
    
    // Update collaborative agents chart
    if (collaborativeChart) {
        // Mock data for collaborative agents
        const agentData = [2, 1, 0, 3, 1]; // sessions per agent
        collaborativeChart.data.datasets[0].data = agentData;
        collaborativeChart.update();
    }
    
    // Update response time chart
    if (responseTimeChart) {
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        const responseTime = stats.avg_response_time ? stats.avg_response_time * 1000 : 0;
        
        // Keep only last 20 points
        if (responseTimeChart.data.labels.length > 20) {
            responseTimeChart.data.labels.shift();
            responseTimeChart.data.datasets[0].data.shift();
        }
        
        responseTimeChart.data.labels.push(timeLabel);
        responseTimeChart.data.datasets[0].data.push(responseTime);
        responseTimeChart.update();
    }
    
    // Update cache chart
    if (cacheChart && cacheStats) {
        const hits = cacheStats.cache_hits || 0;
        const misses = cacheStats.cache_misses || 0;
        
        cacheChart.data.datasets[0].data = [hits, misses];
        cacheChart.update();
    }
}

function showError(message) {
    console.error(message);
    // You could show a toast notification here
}

// Handle API errors gracefully
function handleAPIError(response) {
    if (!response.ok) {
        return {};
    }
    return response.json();
}
