# Enhanced Elephant Deterrence & Monitoring System

A comprehensive web-based system for adaptive elephant deterrence with habituation prevention, real-time monitoring, future predictions, and intelligent decision support for wildlife officers.

## Features

### Core Deterrence System
- **Reinforcement Learning Model**: SARSA-based agent with 5 action strategies
- **Interactive Web Interface**: Modern, responsive dashboard with real-time updates
- **Real-time Predictions**: Input parameters and get instant predictions
- **Sound Playback**: Audio feedback for deterrent actions (with Web Audio API fallback)
- **Habituation Tracking**: Monitor and prevent deterrent habituation over time
- **Visual Analytics**: Multiple charts including Q-values, training metrics, and feature importance
- **Real-Time Simulation**: Run multi-step scenarios to see model behavior
- **Auto Simulation**: Continuous automatic simulation mode
- **Comprehensive Statistics**: Model performance metrics and feature importance rankings
- **Sample Data Generation**: Automatic generation of realistic input scenarios

### Advanced Monitoring & Prediction System (NEW)
- **Real-time Elephant Detection**: Continuous monitoring and detection tracking
- **Unit Health Monitoring**: Track battery, sensors, and operational status of all monitoring units
- **Future Activity Predictions**: AI-powered 24/48/72-hour forecasts of elephant activity
- **Data Collection & Learning**: Historical data storage that improves predictions over time
- **Decision Support System**: Intelligent recommendations and alerts for wildlife officers
- **Advanced Analytics Dashboard**: Comprehensive insights, patterns, and performance metrics
- **Temporal Pattern Analysis**: Hourly and daily activity pattern identification
- **Predictive Maintenance**: Unit health tracking with maintenance recommendations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
./start.sh
```
or
```bash
./run.sh
```

3. Open your browser and navigate to:
```
http://localhost:5001          # Main deterrence system
http://localhost:5001/monitoring  # Monitoring & prediction dashboard
```

**Note:** Port 5001 is used to avoid conflict with macOS AirPlay Receiver on port 5000.

## Usage

### Main Deterrence System (`/`)

1. **Initialize Model**: Click "Initialize Model" to prepare the dataset (5,000 samples)
2. **Train Model**: Click "Train Model" to train the RL agent (100 episodes)
3. **Generate Sample Data**: Click "Generate Sample Data" to populate input fields with random values
4. **Run Prediction**: Adjust parameters and click "Run Prediction" to get action recommendations
5. **Play Sound**: Click "Play Deterrent Sound" to trigger audio feedback

### Monitoring & Prediction Dashboard (`/monitoring`)

1. **View System Overview**: Check total detections, active units, and system health
2. **Monitor Unit Health**: Review battery levels, sensor health, and operational status
3. **Generate Predictions**: Click "Generate 24h Predictions" to see future activity forecasts
4. **Review Recommendations**: Check decision support recommendations and alerts
5. **Analyze Patterns**: Explore hourly patterns, trends, and analytics
6. **Track Detections**: View recent detections and historical data

### Advanced Features

- **Real-Time Simulation**: Click "Run Simulation" to see 10-step scenario progression
- **Auto Simulate**: Enable continuous automatic simulation (runs every 3 seconds)
- **View Statistics**: After training, see comprehensive model performance metrics
- **Feature Importance**: View top 10 most important features for decision-making
- **Training Metrics**: Monitor accuracy, precision, recall, and loss over training episodes

## Model Actions

- **M0: OBSERVE** - Monitor only, no deterrent
- **M1: SPARSE_BIO** - Sparse biological cues
- **M2: DIRECTIONAL** - Directional reinforcement
- **M3: MULTI_SPECTRAL** - Adaptive multi-spectral
- **M4: HUMAN_ALERT** - Escalate to human intervention

## Project Structure

```
semilation/
├── app.py                    # Main Flask backend
├── model.py                  # RL model implementation
├── monitoring_backend.py     # Monitoring & prediction backend
├── requirements.txt          # Python dependencies
├── data/                     # Data storage (auto-created)
│   ├── historical_data.json
│   └── unit_health.json
├── templates/
│   ├── index.html           # Main deterrence interface
│   └── monitoring.html      # Monitoring dashboard
└── static/
    ├── css/
    │   ├── style.css        # Main interface styling
    │   └── monitoring.css   # Monitoring dashboard styling
    ├── js/
    │   ├── main.js          # Main interface logic
    │   └── monitoring.js   # Monitoring dashboard logic
    └── sounds/              # Sound files directory
```

## API Endpoints

### Deterrence System
- `POST /api/initialize` - Initialize the model with dataset
- `POST /api/train` - Train the RL agent (returns training metrics)
- `POST /api/predict` - Make a prediction with input parameters
- `GET /api/generate-sample` - Generate random sample data
- `GET /api/status` - Get model status and habituation score
- `POST /api/sound/<action>` - Trigger sound playback for action
- `GET /api/stats` - Get comprehensive model statistics
- `POST /api/simulate` - Run multi-step simulation scenario

### Monitoring System
- `GET /api/monitoring/status` - Get overall system status
- `GET /api/monitoring/detections` - Retrieve historical detections
- `POST /api/monitoring/detections` - Record new detection
- `GET /api/monitoring/units` - Get health status of all units
- `GET /api/monitoring/units/<unit_id>` - Get specific unit health
- `PUT /api/monitoring/units/<unit_id>` - Update unit health
- `GET /api/monitoring/predictions` - Retrieve predictions
- `POST /api/monitoring/predictions` - Generate new predictions
- `GET /api/monitoring/recommendations` - Get decision support recommendations
- `GET /api/monitoring/analytics` - Get comprehensive analytics

## Technologies

- **Backend**: Flask, Python, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **ML**: scikit-learn, NumPy, Pandas
- **Visualization**: Chart.js
- **Audio**: Web Audio API with MP3 fallback

## Input Parameters

The model accepts 19 input parameters:

- **Time**: `hour_of_day`, `is_night`
- **Environment**: `ambient_noise_db`, `weather_condition`, `wind_speed_kmh`, `temperature_celsius`
- **Elephant**: `elephant_count`, `aggression_level`, `proximity_to_boundary_m`, `movement_speed`
- **Deterrent**: `deterrent_uses_24h`, `days_since_last_use`, `cumulative_exposure_score`, `effectiveness_decay_factor`
- **Risk**: `human_proximity_m`, `crop_value_zone`, `boundary_segment_risk`
- **System**: `sensor_confidence`, `battery_level`

## Output

The model provides:

- **Action**: One of 5 deterrent strategies (M0-M4)
- **Predicted Effectiveness**: Failed (0), Partial (1), or Success (2)
- **Q-Values**: Confidence scores for each action
- **Risk Metrics**: Habituation risk, human risk, threat level
- **Habituation Score**: Current habituation tracking value
- **Cooldown Recommendation**: Whether to pause deterrent usage