// Main JavaScript for Elephant Deterrence System

const API_BASE = '';

// State management
let modelState = {
    initialized: false,
    trained: false,
    currentPrediction: null,
    trainingData: null
};

// Input field definitions
const inputFields = [
    { name: 'hour_of_day', label: 'Hour of Day', type: 'number', min: 0, max: 23, default: 12 },
    { name: 'is_night', label: 'Is Night', type: 'select', options: [{value: 0, label: 'No'}, {value: 1, label: 'Yes'}], default: 0 },
    { name: 'ambient_noise_db', label: 'Ambient Noise (dB)', type: 'number', min: 30, max: 80, default: 45 },
    { name: 'weather_condition', label: 'Weather', type: 'select', options: [{value: 0, label: 'Clear'}, {value: 1, label: 'Cloudy'}, {value: 2, label: 'Storm'}], default: 0 },
    { name: 'wind_speed_kmh', label: 'Wind Speed (km/h)', type: 'number', min: 0, max: 50, default: 10 },
    { name: 'temperature_celsius', label: 'Temperature (°C)', type: 'number', min: 15, max: 40, default: 28 },
    { name: 'elephant_count', label: 'Elephant Count', type: 'number', min: 1, max: 8, default: 2 },
    { name: 'aggression_level', label: 'Aggression Level', type: 'select', options: [{value: 0, label: 'Low'}, {value: 1, label: 'Medium'}, {value: 2, label: 'High'}, {value: 3, label: 'Very High'}], default: 0 },
    { name: 'proximity_to_boundary_m', label: 'Proximity to Boundary (m)', type: 'number', min: 5, max: 200, default: 50 },
    { name: 'movement_speed', label: 'Movement Speed', type: 'select', options: [{value: 0, label: 'Stationary'}, {value: 1, label: 'Slow'}, {value: 2, label: 'Fast'}], default: 0 },
    { name: 'deterrent_uses_24h', label: 'Deterrent Uses (24h)', type: 'number', min: 0, max: 10, default: 2 },
    { name: 'days_since_last_use', label: 'Days Since Last Use', type: 'number', min: 0, max: 20, default: 3 },
    { name: 'cumulative_exposure_score', label: 'Cumulative Exposure', type: 'number', min: 0, max: 15, default: 4 },
    { name: 'effectiveness_decay_factor', label: 'Effectiveness Decay', type: 'number', min: 0, max: 1, step: 0.01, default: 0.8 },
    { name: 'human_proximity_m', label: 'Human Proximity (m)', type: 'number', min: 20, max: 500, default: 100 },
    { name: 'crop_value_zone', label: 'Crop Value Zone', type: 'select', options: [{value: 0, label: 'Low'}, {value: 1, label: 'Medium'}, {value: 2, label: 'High'}], default: 1 },
    { name: 'boundary_segment_risk', label: 'Boundary Risk', type: 'select', options: [{value: 0, label: 'Low'}, {value: 1, label: 'Medium'}, {value: 2, label: 'High'}], default: 1 },
    { name: 'sensor_confidence', label: 'Sensor Confidence', type: 'number', min: 0, max: 1, step: 0.01, default: 0.8 },
    { name: 'battery_level', label: 'Battery Level', type: 'number', min: 0, max: 1, step: 0.01, default: 0.9 }
];

let qValuesChart = null;
let trainingChart = null;
let trainingMetricsChart = null;
let featureImportanceChart = null;
let autoSimulateInterval = null;
let audioContext = null;
let audioUnlocked = false;
let workflowLogs = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeInputs();
    setupEventListeners();
    checkStatus();
    initializeQValuesChart();
    initializeTrainingMetricsChart();
    loadAvailableSounds();
});
document.addEventListener('click', unlockAudio, { once: true });

function unlockAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    audioUnlocked = true;
    addWorkflowLog('🔓 Audio unlocked by user interaction', 'success');
}
function playWebAudioFallback(soundName) {
    try {
        if (!audioUnlocked || !audioContext) {
            addWorkflowLog('⚠️ Audio not unlocked yet', 'warning');
            return;
        }

        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        const soundConfig = {
            observe: { freq: 200, duration: 0.3, type: 'sine' },
            sparse_bio: { freq: 400, duration: 0.5, type: 'sine' },
            directional: { freq: 600, duration: 0.4, type: 'square' },
            multi_spectral: { freq: 800, duration: 0.6, type: 'sawtooth' },
            human_alert: { freq: 1000, duration: 0.8, type: 'square' }
        };

        const config =
            soundConfig[soundName] ||
            soundConfig[soundName.toLowerCase()] ||
            { freq: 500, duration: 0.5, type: 'sine' };

        oscillator.frequency.value = config.freq;
        oscillator.type = config.type;

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(
            0.01,
            audioContext.currentTime + config.duration
        );

        oscillator.start();
        oscillator.stop(audioContext.currentTime + config.duration);

        addWorkflowLog(`✓ Auto sound played: ${soundName}`, 'success');
    } catch (err) {
        addWorkflowLog(`✗ Audio error: ${err.message}`, 'error');
    }
}

async function loadAvailableSounds() {
    try {
        const response = await fetch(`${API_BASE}/api/sounds/list`);
        const data = await response.json();
        
        if (data.success && data.sounds.length > 0) {
            addWorkflowLog(`📁 Found ${data.count} sound files in sounds folder:`, 'info');
            data.sounds.forEach(sound => {
                addWorkflowLog(`  • ${sound.filename} (${(sound.size / 1024).toFixed(1)} KB)`, 'info');
            });
        } else {
            addWorkflowLog(`ℹ️ No sound files found in sounds folder`, 'info');
        }
    } catch (error) {
        console.error('Failed to load sounds list:', error);
    }
}

function initializeInputs() {
    const inputGrid = document.getElementById('input-grid');
    inputGrid.innerHTML = '';
    
    inputFields.forEach(field => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = field.label;
        label.setAttribute('for', field.name);
        
        let input;
        if (field.type === 'select') {
            input = document.createElement('select');
            input.id = field.name;
            field.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                if (opt.value === field.default) option.selected = true;
                input.appendChild(option);
            });
        } else {
            input = document.createElement('input');
            input.type = field.type;
            input.id = field.name;
            input.min = field.min;
            input.max = field.max;
            input.step = field.step || 1;
            input.value = field.default;
        }
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        inputGrid.appendChild(inputGroup);
    });
}

function setupEventListeners() {
    document.getElementById('init-btn').addEventListener('click', initializeModel);
    document.getElementById('train-btn').addEventListener('click', trainModel);
    document.getElementById('generate-btn').addEventListener('click', generateSampleData);
    document.getElementById('predict-btn').addEventListener('click', runPrediction);
    document.getElementById('clear-btn').addEventListener('click', clearInputs);
    document.getElementById('simulate-btn').addEventListener('click', runSimulation);
    document.getElementById('auto-simulate-btn').addEventListener('click', toggleAutoSimulate);
    document.getElementById('clear-logs-btn').addEventListener('click', clearWorkflowLogs);
    document.getElementById('export-logs-btn').addEventListener('click', exportWorkflowLogs);
}

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        updateStatus(data.initialized, data.trained);
        if (data.habituation_score !== undefined) {
            updateHabituationScore(data.habituation_score);
        }
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

function updateStatus(initialized, trained) {
    const indicator = document.getElementById('status-indicator');
    const initBtn = document.getElementById('init-btn');
    const trainBtn = document.getElementById('train-btn');
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    modelState.initialized = initialized;
    modelState.trained = trained;
    
    if (trained) {
        indicator.textContent = 'Trained & Ready';
        indicator.className = 'status-badge trained';
        trainBtn.disabled = false;
        generateBtn.disabled = false;
        predictBtn.disabled = false;
    } else if (initialized) {
        indicator.textContent = 'Initialized';
        indicator.className = 'status-badge initialized';
        trainBtn.disabled = false;
    } else {
        indicator.textContent = 'Not Initialized';
        indicator.className = 'status-badge';
        trainBtn.disabled = true;
        generateBtn.disabled = true;
        predictBtn.disabled = true;
    }
}

function updateHabituationScore(score) {
    const badge = document.getElementById('habituation-score');
    badge.textContent = `Habituation: ${score.toFixed(2)}`;
    
    if (score > 0.6) {
        badge.style.background = '#ef4444';
    } else if (score > 0.3) {
        badge.style.background = '#f59e0b';
    } else {
        badge.style.background = '#10b981';
    }
}

async function initializeModel() {
    showLoading('Initializing model...');
    addWorkflowLog('🚀 Starting model initialization...', 'info');
    addWorkflowLog('📊 Generating synthetic dataset (5,000 samples)...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/api/initialize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_samples: 5000 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addWorkflowLog(`✓ Dataset generated: ${data.data.train_samples} train, ${data.data.val_samples} val, ${data.data.test_samples} test`, 'success');
            addWorkflowLog(`✓ Features extracted: ${data.data.features} total features`, 'success');
            addWorkflowLog('✓ Feature engineering completed (habituation_risk, threat_level, etc.)', 'success');
            addWorkflowLog('✓ Data preprocessing and scaling completed', 'success');
            addWorkflowLog('✓ RL Agent initialized (SARSA with 5 actions)', 'success');
            addWorkflowLog('✓ Model ready for training!', 'success');
            updateStatus(true, false);
            showNotification('Model initialized successfully!', 'success');
        } else {
            addWorkflowLog(`✗ Initialization failed: ${data.message}`, 'error');
            showNotification('Initialization failed: ' + data.message, 'error');
        }
    } catch (error) {
        addWorkflowLog(`✗ Error: ${error.message}`, 'error');
        showNotification('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function trainModel() {
    const modal = document.getElementById('training-modal');
    modal.classList.add('active');
    addWorkflowLog('🎓 Starting RL training (SARSA algorithm)...', 'info');
    addWorkflowLog('⚙️ Training configuration: 100 episodes, learning_rate=0.01, gamma=0.95', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_episodes: 100 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addWorkflowLog('📈 Training progress:', 'info');
            const finalTrainAcc = data.data.train_accuracy[data.data.train_accuracy.length - 1];
            const finalValAcc = data.data.val_accuracy[data.data.val_accuracy.length - 1];
            addWorkflowLog(`  • Final Train Accuracy: ${(finalTrainAcc * 100).toFixed(2)}%`, 'success');
            addWorkflowLog(`  • Final Val Accuracy: ${(finalValAcc * 100).toFixed(2)}%`, 'success');
            addWorkflowLog(`  • Average Reward: ${data.data.rewards[data.data.rewards.length - 1].toFixed(4)}`, 'info');
            addWorkflowLog(`  • Final Loss: ${data.data.losses[data.data.losses.length - 1].toFixed(4)}`, 'info');
            
            if (data.stats) {
                addWorkflowLog('📊 Model Evaluation:', 'info');
                addWorkflowLog(`  • Test Accuracy: ${(data.stats.test_accuracy * 100).toFixed(2)}%`, 'success');
                addWorkflowLog(`  • Test Precision: ${(data.stats.test_precision * 100).toFixed(2)}%`, 'success');
                addWorkflowLog(`  • Test Recall: ${(data.stats.test_recall * 100).toFixed(2)}%`, 'success');
                addWorkflowLog(`  • Test F1-Score: ${(data.stats.test_f1 * 100).toFixed(2)}%`, 'success');
                addWorkflowLog(`  • Top Feature: ${data.stats.top_features[0]?.name || 'N/A'}`, 'info');
            }
            
            addWorkflowLog('✓ Training completed successfully!', 'success');
            
            modelState.trainingData = data.data;
            updateStatus(true, true);
            showNotification('Training completed!', 'success');
            initializeTrainingChart(data.data);
            updateTrainingMetricsChart(data.data);
            
            // Load model stats
            if (data.stats) {
                displayModelStats(data.stats);
                updateFeatureImportanceChart(data.stats.top_features);
            } else {
                loadModelStats();
            }
            
            // Enable simulate button
            document.getElementById('simulate-btn').disabled = false;
            document.getElementById('auto-simulate-btn').disabled = false;
        } else {
            addWorkflowLog(`✗ Training failed: ${data.message}`, 'error');
            showNotification('Training failed: ' + data.message, 'error');
        }
    } catch (error) {
        addWorkflowLog(`✗ Error: ${error.message}`, 'error');
        showNotification('Error: ' + error.message, 'error');
    } finally {
        modal.classList.remove('active');
    }
}

async function generateSampleData() {
    showLoading('Generating sample data...');
    addWorkflowLog('🎲 Generating random sample data...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/api/generate-sample`);
        const data = await response.json();
        
        if (data.success) {
            addWorkflowLog('✓ Sample data generated with realistic distributions', 'success');
            addWorkflowLog(`  • Elephants: ${data.data.elephant_count}, Aggression: ${data.data.aggression_level}`, 'info');
            addWorkflowLog(`  • Proximity: ${data.data.proximity_to_boundary_m.toFixed(1)}m, Weather: ${data.data.weather_condition}`, 'info');
            populateInputs(data.data);
            showNotification('Sample data generated!', 'success');
            return true;
        } else {
            addWorkflowLog(`✗ Generation failed: ${data.message}`, 'error');
            showNotification('Generation failed: ' + data.message, 'error');
            return false;
        }
    } catch (error) {
        addWorkflowLog(`✗ Error: ${error.message}`, 'error');
        showNotification('Error: ' + error.message, 'error');
        return false;
    } finally {
        hideLoading();
    }
}

function populateInputs(data) {
    inputFields.forEach(field => {
        const input = document.getElementById(field.name);
        if (input && data[field.name] !== undefined) {
            input.value = data[field.name];
        }
    });
}

function clearInputs() {
    inputFields.forEach(field => {
        const input = document.getElementById(field.name);
        if (input) {
            input.value = field.default;
        }
    });
}

async function runPrediction() {
    if (!modelState.trained) {
        showNotification('Please train the model first!', 'error');
        return;
    }
    
    showLoading('Running prediction...');
    addWorkflowLog('🔮 Starting prediction workflow...', 'info');
    
    try {
        const inputData = {};
        inputFields.forEach(field => {
            const input = document.getElementById(field.name);
            if (input) {
                const value = field.type === 'number' ? parseFloat(input.value) : parseInt(input.value);
                inputData[field.name] = value;
            }
        });
        
        addWorkflowLog('📥 Input parameters collected:', 'info');
        addWorkflowLog(`  • Elephants: ${inputData.elephant_count}, Aggression: ${inputData.aggression_level}`, 'info');
        addWorkflowLog(`  • Proximity: ${inputData.proximity_to_boundary_m}m, Threat Level: Calculating...`, 'info');
        
        addWorkflowLog('🧠 RL Agent processing state...', 'info');
        addWorkflowLog('  • Computing Q-values for all 5 actions', 'info');
        addWorkflowLog('  • Applying epsilon-greedy policy (exploitation mode)', 'info');
        
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            const pred = data.prediction;
            addWorkflowLog('✓ Q-values computed:', 'success');
            pred.q_values.forEach((q, i) => {
                const actions = ['OBSERVE', 'SPARSE_BIO', 'DIRECTIONAL', 'MULTI_SPECTRAL', 'HUMAN_ALERT'];
                addWorkflowLog(`  • ${actions[i]}: ${q.toFixed(4)}`, 'info');
            });
            
            addWorkflowLog(`🎯 Selected Action: ${pred.action_name} (Action ${pred.action})`, 'success');
            addWorkflowLog(`  • Predicted Effectiveness: ${pred.class_name} (Class ${pred.predicted_class})`, 'info');
            addWorkflowLog(`  • Confidence: ${(pred.confidence * 100).toFixed(2)}%`, 'info');
            addWorkflowLog(`  • Habituation Risk: ${(pred.habituation_risk * 100).toFixed(2)}%`, 'info');
            addWorkflowLog(`  • Human Risk: ${(pred.human_risk * 100).toFixed(2)}%`, 'info');
            addWorkflowLog(`  • Threat Level: ${(pred.threat_level * 100).toFixed(2)}%`, 'info');
            
            if (data.habituation) {
                addWorkflowLog(`📊 Habituation Score: ${(data.habituation.score * 100).toFixed(2)}%`, 'info');
                if (data.habituation.cooldown_recommended) {
                    addWorkflowLog('⚠️ Cooldown recommended - habituation risk high!', 'warning');
                }
            }
            
            addWorkflowLog('✓ Prediction completed successfully!', 'success');
            
            modelState.currentPrediction = data.prediction;
            displayResults(data.prediction, data.habituation);
            updateMetrics(data.prediction);
            updateQValuesChart(data.prediction.q_values);
            highlightAction(data.prediction.action);
            
            // Auto-play sound after prediction
            if (data.prediction.action !== 0) {
                addWorkflowLog(`🔊 Auto-playing sound for action: ${data.prediction.action_name}`, 'sound');
                setTimeout(() => {
                    playSoundForAction(data.prediction.action, data.prediction);
                }, 500);
            }
            
            if (data.habituation) {
                updateHabituationScore(data.habituation.score);
            }
            
            showNotification('Prediction completed!', 'success');
        } else {
            addWorkflowLog(`✗ Prediction failed: ${data.message}`, 'error');
            showNotification('Prediction failed: ' + data.message, 'error');
        }
    } catch (error) {
        addWorkflowLog(`✗ Error: ${error.message}`, 'error');
        showNotification('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displayResults(prediction, habituation) {
    const container = document.getElementById('results-container');
    
    const resultsHTML = `
        <div class="results-content">
            <div class="result-item">
                <span class="result-label">Selected Action:</span>
                <span class="result-value">${prediction.action_name}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Predicted Effectiveness:</span>
                <span class="result-value">${prediction.class_name}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Confidence:</span>
                <span class="result-value">${(prediction.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Habituation Risk:</span>
                <span class="result-value">${(prediction.habituation_risk * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Human Risk:</span>
                <span class="result-value">${(prediction.human_risk * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Threat Level:</span>
                <span class="result-value">${(prediction.threat_level * 100).toFixed(2)}%</span>
            </div>
            ${habituation ? `
            <div class="result-item">
                <span class="result-label">Habituation Score:</span>
                <span class="result-value">${(habituation.score * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-label">Cooldown Recommended:</span>
                <span class="result-value">${habituation.cooldown_recommended ? 'Yes' : 'No'}</span>
            </div>
            ` : ''}
        </div>
    `;
    
    container.innerHTML = resultsHTML;
}

function updateMetrics(prediction) {
    document.getElementById('metric-class').textContent = prediction.class_name;
    document.getElementById('metric-confidence').textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    document.getElementById('metric-threat').textContent = `${(prediction.threat_level * 100).toFixed(1)}%`;
    document.getElementById('metric-human-risk').textContent = `${(prediction.human_risk * 100).toFixed(1)}%`;
}

function highlightAction(action) {
    document.querySelectorAll('.action-item').forEach(item => {
        item.classList.remove('selected');
        if (parseInt(item.dataset.action) === action) {
            item.classList.add('selected');
        }
    });
}

function initializeQValuesChart() {
    const ctx = document.getElementById('qvalues-chart');
    if (!ctx) return;
    
    qValuesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['OBSERVE', 'SPARSE_BIO', 'DIRECTIONAL', 'MULTI_SPECTRAL', 'HUMAN_ALERT'],
            datasets: [{
                label: 'Q-Values',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(37, 99, 235, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(37, 99, 235, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(139, 92, 246, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#334155'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function updateQValuesChart(qValues) {
    if (qValuesChart) {
        qValuesChart.data.datasets[0].data = qValues;
        qValuesChart.update();
    }
}

function initializeTrainingChart(data) {
    const ctx = document.getElementById('training-chart');
    if (!ctx || !data) return;
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.train_accuracy.map((_, i) => i + 1),
            datasets: [
                {
                    label: 'Train Accuracy',
                    data: data.train_accuracy,
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Val Accuracy',
                    data: data.val_accuracy,
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function initializeTrainingMetricsChart() {
    const ctx = document.getElementById('training-metrics-chart');
    if (!ctx) return;
    
    trainingMetricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Accuracy',
                    data: [],
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Val Accuracy',
                    data: [],
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Loss',
                    data: [],
                    borderColor: 'rgba(239, 68, 68, 1)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    position: 'left',
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: { color: '#94a3b8' },
                    grid: { drawOnChartArea: false }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

function updateTrainingMetricsChart(data) {
    if (!trainingMetricsChart || !data) return;
    
    const episodes = data.train_accuracy.map((_, i) => i + 1);
    
    trainingMetricsChart.data.labels = episodes;
    trainingMetricsChart.data.datasets[0].data = data.train_accuracy;
    trainingMetricsChart.data.datasets[1].data = data.val_accuracy;
    trainingMetricsChart.data.datasets[2].data = data.losses || [];
    trainingMetricsChart.update();
}

async function playSound() {
    if (!modelState.currentPrediction) {
        addWorkflowLog('⚠️ No prediction available. Run prediction first!', 'warning');
        return;
    }
    
    const action = modelState.currentPrediction.action;
    const actionName = modelState.currentPrediction.action_name;
    
    await playSoundForAction(action, modelState.currentPrediction);
}

async function playSoundForAction(action, prediction = null) {
    const actionNames = ['OBSERVE', 'SPARSE_BIO', 'DIRECTIONAL', 'MULTI_SPECTRAL', 'HUMAN_ALERT'];
    const actionName = prediction?.action_name || actionNames[action] || 'UNKNOWN';
    const soundStatus = document.getElementById('sound-status');
    
    addWorkflowLog(`🔊 Sound playback requested for: ${actionName} (Action ${action})`, 'sound');
    if (soundStatus) {
        soundStatus.textContent = `Playing sound for ${actionName}...`;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/sound/${action}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.sound_path) {
                addWorkflowLog(`✓ Sound file found: ${data.sound}`, 'success');
                addWorkflowLog(`  • Using file: ${data.sound_path}`, 'info');
                // Play the actual sound file
                playAudioFeedback(data);
                if (soundStatus) {
                    soundStatus.textContent = `✓ Playing: ${data.sound}`;
                }
            } else if (data.sound === null) {
                addWorkflowLog(`ℹ️ No sound for action: ${actionName} (OBSERVE mode)`, 'info');
                if (soundStatus) {
                    soundStatus.textContent = `No sound for ${actionName}`;
                }
            } else {
                addWorkflowLog(`⚠️ Sound file not found, using fallback`, 'warning');
                playWebAudioFallback(data.sound || actionName.toLowerCase());
                if (soundStatus) {
                    soundStatus.textContent = `✓ Playing fallback sound`;
                }
            }
            
            if (soundStatus) {
                setTimeout(() => {
                    soundStatus.textContent = '';
                }, 5000);
            }
        } else {
            addWorkflowLog(`✗ Sound error: ${data.message}`, 'error');
            if (soundStatus) {
                soundStatus.textContent = `Error: ${data.message}`;
            }
        }
    } catch (error) {
        addWorkflowLog(`✗ Sound API error: ${error.message}`, 'error');
        if (soundStatus) {
            soundStatus.textContent = 'Error playing sound';
        }
        console.error('Sound error:', error);
    }
}

function playAudioFeedback(soundData) {
    // soundData can be a filename string or an object with sound_path
    let soundPath, soundName;
    
    if (typeof soundData === 'string') {
        // Old format - just filename
        soundName = soundData;
        soundPath = `/static/sounds/${soundName}`;
    } else if (soundData && soundData.sound_path) {
        // New format - object with sound_path
        soundPath = soundData.sound_path;
        soundName = soundData.sound || soundData.sound_path.split('/').pop();
    } else {
        addWorkflowLog(`⚠️ Invalid sound data format`, 'warning');
        return;
    }
    
    addWorkflowLog(`🔊 Attempting to play sound: ${soundName}`, 'sound');
    addWorkflowLog(`  • Sound path: ${soundPath}`, 'info');
    
    const audio = new Audio(soundPath);
    
    // Set up event handlers
    audio.addEventListener('loadeddata', () => {
        addWorkflowLog(`✓ Sound file loaded: ${soundName}`, 'success');
    }, { once: true });
    
    audio.addEventListener('error', (e) => {
        addWorkflowLog(`⚠️ Sound file error: ${e.message || 'File not found'}, using Web Audio API fallback`, 'warning');
        playWebAudioFallback(soundName);
    }, { once: true });
    
    audio.addEventListener('ended', () => {
        addWorkflowLog(`✓ Sound playback completed: ${soundName}`, 'info');
    }, { once: true });
    
    // Try to play the audio file
    audio.play().then(() => {
        addWorkflowLog(`✓ Sound playing: ${soundName}`, 'success');
    }).catch((error) => {
        addWorkflowLog(`⚠️ Audio play failed: ${error.message}, using fallback`, 'warning');
        playWebAudioFallback(soundName);
    });
}

function playWebAudioFallback(soundName) {
    try {
        addWorkflowLog(`🔊 Using Web Audio API fallback for: ${soundName}`, 'sound');
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Different frequencies and patterns for different actions
        const soundConfig = {
            'observe': { freq: 200, duration: 0.3, type: 'sine' },
            'sparse_bio': { freq: 400, duration: 0.5, type: 'sine' },
            'bees-82424.mp3': { freq: 400, duration: 0.5, type: 'sine' },
            'bees-swarming-98657.mp3': { freq: 400, duration: 0.5, type: 'sine' },
            'directional': { freq: 600, duration: 0.4, type: 'square' },
            'tiger-light-roar-t-293716.mp3': { freq: 600, duration: 0.4, type: 'square' },
            'multi_spectral': { freq: 800, duration: 0.6, type: 'sawtooth' },
            'tiger-roar-104166.mp3': { freq: 800, duration: 0.6, type: 'sawtooth' },
            'tiger-roar-399277.mp3': { freq: 800, duration: 0.6, type: 'sawtooth' },
            'human_alert': { freq: 1000, duration: 0.8, type: 'square' },
            'tiger-roar-loudly-193229.mp3': { freq: 1000, duration: 0.8, type: 'square' },
            'tiger-attack-195840.mp3': { freq: 1000, duration: 0.8, type: 'square' }
        };
        
        // Try to match by filename or action name
        const config = soundConfig[soundName] || 
                      soundConfig[soundName.toLowerCase()] || 
                      { freq: 500, duration: 0.5, type: 'sine' };
        
        oscillator.frequency.value = config.freq;
        oscillator.type = config.type;
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + config.duration);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + config.duration);
        
        addWorkflowLog(`✓ Web Audio played: ${soundName} (${config.freq}Hz, ${config.type})`, 'success');
    } catch (error) {
        addWorkflowLog(`✗ Sound playback error: ${error.message}`, 'error');
        console.error('Web Audio error:', error);
    }
}

async function loadModelStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        if (data.success) {
            displayModelStats(data.stats);
            updateFeatureImportanceChart(data.stats.top_features);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

function displayModelStats(stats) {
    const container = document.getElementById('model-stats');
    
    const statsHTML = `
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-label">Test Accuracy</div>
                <div class="stat-value">${(stats.test_accuracy * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Precision</div>
                <div class="stat-value">${(stats.test_precision * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Recall</div>
                <div class="stat-value">${(stats.test_recall * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">F1-Score</div>
                <div class="stat-value">${(stats.test_f1 * 100).toFixed(2)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Features</div>
                <div class="stat-value">${stats.n_features}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Actions</div>
                <div class="stat-value">${stats.n_actions}</div>
            </div>
        </div>
    `;
    
    container.innerHTML = statsHTML;
}

function updateFeatureImportanceChart(features) {
    if (!features || features.length === 0) return;
    
    const ctx = document.getElementById('feature-importance-chart');
    if (!ctx) return;
    
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    const labels = features.map(f => f.name);
    const data = features.map(f => f.importance);
    
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance',
                data: data,
                backgroundColor: 'rgba(37, 99, 235, 0.8)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function runSimulation() {
    if (!modelState.trained) {
        showNotification('Please train the model first!', 'error');
        return;
    }
    
    showLoading('Running simulation...');
    addWorkflowLog('🎮 Starting multi-step simulation (10 scenarios)...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_steps: 10 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addWorkflowLog(`✓ Simulation completed: ${data.n_steps} steps processed`, 'success');
            addWorkflowLog('📊 Simulation Summary:', 'info');
            
            const scenarios = data.scenarios;
            const actionCounts = {};
            const effectivenessCounts = {0: 0, 1: 0, 2: 0};
            
            scenarios.forEach(scenario => {
                const actionName = scenario.prediction.action_name;
                actionCounts[actionName] = (actionCounts[actionName] || 0) + 1;
                effectivenessCounts[scenario.prediction.predicted_class]++;
            });
            
            addWorkflowLog('  • Actions selected:', 'info');
            Object.entries(actionCounts).forEach(([action, count]) => {
                addWorkflowLog(`    - ${action}: ${count} times`, 'info');
            });
            
            addWorkflowLog('  • Effectiveness distribution:', 'info');
            addWorkflowLog(`    - Failed: ${effectivenessCounts[0]}, Partial: ${effectivenessCounts[1]}, Success: ${effectivenessCounts[2]}`, 'info');
            
            const finalHabituation = scenarios[scenarios.length - 1].habituation_score;
            addWorkflowLog(`  • Final Habituation Score: ${(finalHabituation * 100).toFixed(2)}%`, 'info');
            
            if (scenarios.some(s => s.cooldown_recommended)) {
                addWorkflowLog('⚠️ Cooldown recommended in some scenarios', 'warning');
            }
            
            displaySimulationResults(data.scenarios);
            showNotification(`Simulation completed: ${data.n_steps} steps`, 'success');
            
            // Auto-play sound for each scenario that has a sound
            if (data.scenarios && data.scenarios.length > 0) {
                addWorkflowLog(`🔊 Auto-playing sounds for ${data.scenarios.length} scenarios...`, 'sound');
                
                // Play sounds for each scenario with a delay between them
                data.scenarios.forEach((scenario, index) => {
                    const action = scenario.prediction.action;
                    const actionName = scenario.prediction.action_name;
                    
                    // Only play sound if action is not OBSERVE (action 0)
                    if (action !== 0) {
                        setTimeout(() => {
                            addWorkflowLog(`🔊 Auto-playing sound for step ${index + 1}: ${actionName}`, 'sound');
                            modelState.currentPrediction = scenario.prediction;
                            playSoundForAction(action, scenario.prediction);
                        }, (index + 1) * 2000); // 2 second delay between each sound
                    }
                });
            }
            
            // Notify monitoring dashboard to refresh if it's open
            if (data.monitoring_updated) {
                addWorkflowLog('📊 Monitoring system updated with simulation data', 'success');
                // Trigger custom event for monitoring dashboard
                window.dispatchEvent(new CustomEvent('simulationCompleted', { 
                    detail: { scenarios: data.scenarios } 
                }));
            }
        } else {
            addWorkflowLog(`✗ Simulation failed: ${data.message}`, 'error');
            showNotification('Simulation failed: ' + data.message, 'error');
        }
    } catch (error) {
        addWorkflowLog(`✗ Error: ${error.message}`, 'error');
        showNotification('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function displaySimulationResults(scenarios) {
    const container = document.getElementById('simulation-results');
    
    let html = '<div class="simulation-steps">';
    
    scenarios.forEach((scenario, idx) => {
        const actionNames = ['OBSERVE', 'SPARSE_BIO', 'DIRECTIONAL', 'MULTI_SPECTRAL', 'HUMAN_ALERT'];
        const classNames = ['Failed', 'Partial', 'Success'];
        const actionName = actionNames[scenario.prediction.action];
        const className = classNames[scenario.prediction.predicted_class];
        
        html += `
            <div class="simulation-step">
                <div class="step-header">
                    <span class="step-number">Step ${scenario.step}</span>
                    <span class="step-action">${actionName}</span>
                    <span class="step-class ${className.toLowerCase()}">${className}</span>
                </div>
                <div class="step-details">
                    <div class="detail-item">
                        <span>Elephants:</span> ${scenario.input.elephant_count}
                    </div>
                    <div class="detail-item">
                        <span>Threat:</span> ${(scenario.prediction.threat_level * 100).toFixed(1)}%
                    </div>
                    <div class="detail-item">
                        <span>Habituation:</span> ${(scenario.habituation_score * 100).toFixed(1)}%
                    </div>
                    ${scenario.cooldown_recommended ? '<div class="cooldown-warning">⚠️ Cooldown Recommended</div>' : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function toggleAutoSimulate() {
    const btn = document.getElementById('auto-simulate-btn');
    
    if (autoSimulateInterval) {
        clearInterval(autoSimulateInterval);
        autoSimulateInterval = null;
        btn.textContent = '🔄 Auto Simulate';
        btn.classList.remove('active');
        showNotification('Auto simulation stopped', 'info');
    } else {
        if (!modelState.trained) {
            showNotification('Please train the model first!', 'error');
            return;
        }
        
        autoSimulateInterval = setInterval(async () => {
            try {
                const success = await generateSampleData();
                if (success) {
                    setTimeout(() => {
                        runPrediction();
                    }, 500);
                }
            } catch (error) {
                console.error('Auto simulate error:', error);
            }
        }, 3000);
        
        btn.textContent = '⏸️ Stop Auto Simulate';
        btn.classList.add('active');
        showNotification('Auto simulation started (every 3s)', 'success');
    }
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    overlay.querySelector('p').textContent = message;
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('active');
}

function showNotification(message, type = 'info') {
    // Simple notification system
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
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Workflow Logging Functions
function addWorkflowLog(message, type = 'info') {
    const logContainer = document.getElementById('workflow-log');
    if (!logContainer) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    
    const timeSpan = document.createElement('span');
    timeSpan.className = 'log-time';
    timeSpan.textContent = `[${timestamp}]`;
    
    const messageSpan = document.createElement('span');
    messageSpan.className = 'log-message';
    messageSpan.textContent = message;
    
    logEntry.appendChild(timeSpan);
    logEntry.appendChild(messageSpan);
    
    logContainer.appendChild(logEntry);
    
    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
    
    // Store in array for export
    workflowLogs.push({
        timestamp: timestamp,
        type: type,
        message: message
    });
    
    // Keep only last 500 logs
    if (workflowLogs.length > 500) {
        workflowLogs.shift();
    }
}

function clearWorkflowLogs() {
    const logContainer = document.getElementById('workflow-log');
    if (logContainer) {
        logContainer.innerHTML = '<div class="log-entry info"><span class="log-time">[System]</span><span class="log-message">Logs cleared.</span></div>';
        workflowLogs = [];
        addWorkflowLog('Logs cleared by user', 'info');
    }
}

function exportWorkflowLogs() {
    if (workflowLogs.length === 0) {
        showNotification('No logs to export', 'info');
        return;
    }
    
    const logText = workflowLogs.map(log => 
        `[${log.timestamp}] [${log.type.toUpperCase()}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow-logs-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    addWorkflowLog('Logs exported successfully', 'success');
    showNotification('Logs exported!', 'success');
}
