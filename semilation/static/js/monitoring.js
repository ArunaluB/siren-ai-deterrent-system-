// Monitoring Dashboard JavaScript
const API_BASE = '/api/monitoring';

// Chart instances
let predictionsChart = null;
let trendsChart = null;
let hourlyPatternsChart = null;

// State
let updateInterval = null;
let predictionsData = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupEventListeners();
    loadDashboardData();
    startAutoRefresh();
    
    // Listen for simulation completion events from main system
    window.addEventListener('simulationCompleted', (event) => {
        console.log('Simulation completed, refreshing monitoring dashboard...');
        // Refresh all data when simulation completes
        setTimeout(() => {
            loadDashboardData();
            showNotification('Monitoring dashboard updated with simulation data', 'success');
        }, 500);
    });
});

function setupEventListeners() {
    document.getElementById('generate-predictions-btn').addEventListener('click', generatePredictions);
    document.getElementById('prediction-hours').addEventListener('change', generatePredictions);
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function initializeCharts() {
    // Predictions Chart
    const predictionsCtx = document.getElementById('predictions-chart');
    if (predictionsCtx) {
        predictionsChart = new Chart(predictionsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Predicted Elephant Count',
                    data: [],
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }
    
    // Trends Chart
    const trendsCtx = document.getElementById('trends-chart');
    if (trendsCtx) {
        trendsChart = new Chart(trendsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Elephant Activity',
                    data: [],
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }
    
    // Hourly Patterns Chart
    const hourlyCtx = document.getElementById('hourly-patterns-chart');
    if (hourlyCtx) {
        hourlyPatternsChart = new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Average Activity',
                    data: Array(24).fill(0),
                    backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    },
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: '#334155' }
                    }
                }
            }
        });
    }
}

async function loadDashboardData() {
    showLoading();
    
    try {
        await Promise.all([
            loadSystemStatus(),
            loadUnitHealth(),
            loadRecentDetections(),
            loadRecommendations(),
            loadAnalytics()
        ]);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    } finally {
        hideLoading();
    }
}

async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.success) {
            const status = data.status;
            document.getElementById('total-detections').textContent = status.total_detections;
            document.getElementById('active-units').textContent = `${status.active_units}/${status.total_units}`;
            document.getElementById('system-health').textContent = `${(status.system_health * 100).toFixed(0)}%`;
            document.getElementById('last-update').textContent = `Last Update: ${new Date().toLocaleTimeString()}`;
        }
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

async function loadUnitHealth() {
    try {
        const response = await fetch(`${API_BASE}/units`);
        const data = await response.json();
        
        if (data.success) {
            displayUnits(data.units);
        }
    } catch (error) {
        console.error('Error loading unit health:', error);
    }
}

function displayUnits(units) {
    const container = document.getElementById('unit-grid');
    container.innerHTML = '';
    
    units.forEach(unit => {
        const unitCard = document.createElement('div');
        unitCard.className = 'unit-card';
        
        // Determine status class
        if (unit.battery_level < 0.3 || unit.sensor_health < 0.7) {
            unitCard.classList.add('critical');
        } else if (unit.battery_level < 0.5 || unit.sensor_health < 0.8) {
            unitCard.classList.add('warning');
        }
        
        const batteryClass = unit.battery_level < 0.3 ? 'critical' : 
                           unit.battery_level < 0.5 ? 'warning' : '';
        
        unitCard.innerHTML = `
            <div class="unit-header">
                <span class="unit-id">${unit.unit_id}</span>
                <span class="unit-status ${unit.status}">${unit.status}</span>
            </div>
            <div class="unit-metrics">
                <div class="unit-metric">
                    <span class="metric-label">Battery</span>
                    <span class="metric-value">${(unit.battery_level * 100).toFixed(0)}%</span>
                </div>
                <div class="battery-bar">
                    <div class="battery-fill ${batteryClass}" style="width: ${unit.battery_level * 100}%"></div>
                </div>
                <div class="unit-metric">
                    <span class="metric-label">Sensor Health</span>
                    <span class="metric-value">${(unit.sensor_health * 100).toFixed(0)}%</span>
                </div>
                <div class="unit-metric">
                    <span class="metric-label">Detections (24h)</span>
                    <span class="metric-value">${unit.detections_24h}</span>
                </div>
                <div class="unit-metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">${unit.uptime_percentage.toFixed(1)}%</span>
                </div>
            </div>
        `;
        
        container.appendChild(unitCard);
    });
}

async function loadRecentDetections() {
    try {
        const response = await fetch(`${API_BASE}/detections?days=1`);
        const data = await response.json();
        
        if (data.success) {
            displayDetections(data.detections.slice(-10).reverse());
            updateTrendsChart(data.detections);
        }
    } catch (error) {
        console.error('Error loading detections:', error);
    }
}

function displayDetections(detections) {
    const container = document.getElementById('detections-list');
    
    if (detections.length === 0) {
        container.innerHTML = '<div class="empty-state">No detections yet</div>';
        return;
    }
    
    container.innerHTML = detections.map(detection => {
        const time = new Date(detection.timestamp).toLocaleString();
        const count = detection.elephant_count || 0;
        const aggression = detection.aggression_level || 0;
        
        return `
            <div class="detection-item">
                <div class="detection-info">
                    <div class="detection-time">${time}</div>
                    <div class="detection-details">
                        ${count} elephant(s) detected | Aggression: ${aggression}/3
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function updateTrendsChart(detections) {
    if (!trendsChart || detections.length === 0) return;
    
    // Group by date
    const dailyData = {};
    detections.forEach(d => {
        const date = new Date(d.timestamp).toLocaleDateString();
        if (!dailyData[date]) {
            dailyData[date] = { count: 0, total: 0 };
        }
        dailyData[date].total += d.elephant_count || 0;
        dailyData[date].count += 1;
    });
    
    const labels = Object.keys(dailyData).sort();
    const data = labels.map(date => dailyData[date].total / dailyData[date].count);
    
    trendsChart.data.labels = labels;
    trendsChart.data.datasets[0].data = data;
    trendsChart.update();
}

async function generatePredictions() {
    showLoading('Generating predictions...');
    
    try {
        const nHours = parseInt(document.getElementById('prediction-hours').value);
        const response = await fetch(`${API_BASE}/predictions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_hours: nHours })
        });
        
        const data = await response.json();
        
        if (data.success) {
            predictionsData = data.predictions;
            displayPredictions(data.predictions);
            updatePredictionSummary(data.predictions);
        }
    } catch (error) {
        console.error('Error generating predictions:', error);
    } finally {
        hideLoading();
    }
}

function displayPredictions(predictions) {
    if (!predictionsChart || !predictions) return;
    
    const labels = predictions.map(p => {
        const date = new Date(p.timestamp);
        return `${date.getHours()}:00`;
    });
    
    const data = predictions.map(p => p.predicted_elephant_count);
    
    predictionsChart.data.labels = labels;
    predictionsChart.data.datasets[0].data = data;
    predictionsChart.update();
}

function updatePredictionSummary(predictions) {
    const container = document.getElementById('prediction-summary');
    
    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p>No predictions available</p>';
        return;
    }
    
    const avg = predictions.reduce((sum, p) => sum + p.predicted_elephant_count, 0) / predictions.length;
    const max = Math.max(...predictions.map(p => p.predicted_elephant_count));
    const min = Math.min(...predictions.map(p => p.predicted_elephant_count));
    
    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 5px;">Average</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">${avg.toFixed(1)}</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 5px;">Peak</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">${max.toFixed(1)}</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 5px;">Minimum</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">${min.toFixed(1)}</div>
            </div>
        </div>
    `;
}

async function loadRecommendations() {
    try {
        const response = await fetch(`${API_BASE}/recommendations`);
        const data = await response.json();
        
        if (data.success) {
            displayRecommendations(data);
            displayAlerts(data.alerts);
            updateActiveAlertsCount(data.alerts);
        }
    } catch (error) {
        console.error('Error loading recommendations:', error);
    }
}

function displayRecommendations(data) {
    const container = document.getElementById('recommendations-list');
    const priorityBadge = document.getElementById('priority-badge');
    const priorityValue = document.getElementById('priority-value');
    
    // Update priority badge
    priorityBadge.className = `priority-badge ${data.overall_priority}`;
    priorityValue.textContent = data.overall_priority.toUpperCase();
    
    if (!data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = '<div class="empty-state">No recommendations at this time</div>';
        return;
    }
    
    container.innerHTML = data.recommendations.map(rec => `
        <div class="recommendation-item ${rec.priority}">
            <div class="recommendation-header">
                <span class="recommendation-type">${rec.type.replace(/_/g, ' ').toUpperCase()}</span>
                <span class="recommendation-priority ${rec.priority}">${rec.priority.toUpperCase()}</span>
            </div>
            <div class="recommendation-message">${rec.message}</div>
            ${rec.action ? `<div class="recommendation-action">💡 ${rec.action}</div>` : ''}
        </div>
    `).join('');
}

function displayAlerts(alerts) {
    const container = document.getElementById('alerts-list');
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No active alerts</div>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => `
        <div class="alert-item">
            <div class="alert-content">
                <div class="alert-type">${alert.type.replace(/_/g, ' ').toUpperCase()}</div>
                <div class="alert-message">${alert.message}</div>
                ${alert.unit ? `<div class="alert-unit">Unit: ${alert.unit}</div>` : ''}
            </div>
        </div>
    `).join('');
}

function updateActiveAlertsCount(alerts) {
    document.getElementById('active-alerts').textContent = alerts ? alerts.length : 0;
}

async function loadAnalytics() {
    try {
        const response = await fetch(`${API_BASE}/analytics`);
        const data = await response.json();
        
        if (data.success) {
            displayAnalytics(data.analytics);
            updateHourlyPatterns(data.analytics);
        }
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

function displayAnalytics(analytics) {
    // Overview tab
    const overviewContainer = document.getElementById('analytics-overview');
    if (overviewContainer) {
        overviewContainer.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div class="insight-card">
                    <div class="insight-label">Total Detections</div>
                    <div class="insight-value">${analytics.overview?.total_detections || 0}</div>
                </div>
                <div class="insight-card">
                    <div class="insight-label">Average Elephant Count</div>
                    <div class="insight-value">${analytics.elephant_activity?.avg_count?.toFixed(1) || 0}</div>
                </div>
                <div class="insight-card">
                    <div class="insight-label">Peak Activity</div>
                    <div class="insight-value">${analytics.elephant_activity?.max_count || 0}</div>
                </div>
            </div>
        `;
    }
    
    // Patterns tab
    const patternsContainer = document.getElementById('analytics-patterns');
    if (patternsContainer && analytics.temporal_patterns?.hourly) {
        const hourly = analytics.temporal_patterns.hourly;
        patternsContainer.innerHTML = `
            <h3 style="margin-bottom: 15px;">Hourly Activity Distribution</h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                ${Object.entries(hourly).map(([hour, value]) => `
                    <div class="insight-card">
                        <div class="insight-label">${hour}:00</div>
                        <div class="insight-value">${value.toFixed(1)}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    // Units tab
    const unitsContainer = document.getElementById('analytics-units');
    if (unitsContainer && analytics.unit_performance) {
        unitsContainer.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
                ${Object.entries(analytics.unit_performance).map(([unitId, stats]) => `
                    <div class="insight-card">
                        <div class="insight-label">${unitId}</div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Detections:</span>
                                <span style="font-weight: 600;">${stats.detections}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>Uptime:</span>
                                <span style="font-weight: 600;">${stats.uptime.toFixed(1)}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Battery:</span>
                                <span style="font-weight: 600;">${(stats.battery * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
}

function updateHourlyPatterns(analytics) {
    if (!hourlyPatternsChart || !analytics.temporal_patterns?.hourly) return;
    
    const hourly = analytics.temporal_patterns.hourly;
    const data = Array.from({length: 24}, (_, i) => hourly[i] || 0);
    
    hourlyPatternsChart.data.datasets[0].data = data;
    hourlyPatternsChart.update();
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
}

function startAutoRefresh() {
    // Refresh every 30 seconds
    updateInterval = setInterval(() => {
        loadSystemStatus();
        loadUnitHealth();
        loadRecentDetections();
        loadRecommendations();
    }, 30000);
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.querySelector('p').textContent = message;
        overlay.classList.add('active');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#2563eb'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 3000;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
