// Dashboard JavaScript
let categoryChart = null;
let cacheChart = null;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    // Update every 5 seconds
    setInterval(updateDashboard, 5000);
});

async function initializeDashboard() {
    await updateDashboard();
    initializeCharts();
}

async function updateDashboard() {
    try {
        // Load statistics
        const [statsResponse, healthResponse] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/health')
        ]);
        
        const stats = await statsResponse.json();
        const health = await healthResponse.json();
        
        // Update metrics cards
        updateMetricsCards(stats);
        
        // Update system health
        updateSystemHealth(health);
        
        // Update charts
        updateCharts(stats);
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
        showError('Failed to load dashboard data');
    }
}

function updateMetricsCards(stats) {
    document.getElementById('totalQueries').textContent = stats.total_queries || 0;
    
    const successRate = stats.total_queries > 0 ? 
        ((stats.successful_routes / stats.total_queries) * 100).toFixed(1) : 0;
    document.getElementById('successRate').textContent = successRate + '%';
    
    document.getElementById('activeAgents').textContent = stats.active_agents || 0;
    
    const avgResponse = stats.avg_response_time ? 
        (stats.avg_response_time * 1000).toFixed(0) : 0;
    document.getElementById('avgResponse').textContent = avgResponse + 'ms';
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
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF',
                    '#FF9F40',
                    '#FF6384',
                    '#C9CBCF',
                    '#4BC0C0',
                    '#FF6384'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
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
                label: 'Count',
                data: [0, 0],
                backgroundColor: ['#36A2EB', '#FF6384']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updateCharts(stats) {
    // Update category chart
    if (categoryChart && stats.category_distribution) {
        const categories = Object.keys(stats.category_distribution);
        const counts = Object.values(stats.category_distribution);
        
        categoryChart.data.labels = categories;
        categoryChart.data.datasets[0].data = counts;
        categoryChart.update();
    }
    
    // Update cache chart
    if (cacheChart) {
        cacheChart.data.datasets[0].data = [
            stats.cache_hits || 0,
            stats.cache_misses || 0
        ];
        cacheChart.update();
    }
}

function showError(message) {
    const errorHtml = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.querySelector('main').insertAdjacentHTML('afterbegin', errorHtml);
}
