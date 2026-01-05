"""
Enhanced Monitoring and Prediction Backend
==========================================
Advanced system for elephant detection, unit health monitoring, 
future predictions, and decision support for wildlife officers.
"""

from flask import render_template, request, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model import ElephantDeterrenceModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# This module provides functions and classes for monitoring
# Routes will be registered in the main app.py

# Data storage paths
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
HISTORICAL_DATA_FILE = DATA_DIR / 'historical_data.json'
UNIT_HEALTH_FILE = DATA_DIR / 'unit_health.json'
PREDICTIONS_FILE = DATA_DIR / 'predictions.json'

# Initialize models
deterrence_model = ElephantDeterrenceModel()
prediction_models = {}
unit_health_data = {}
historical_data = []

# ============================================================================
# DATA STORAGE & RETRIEVAL
# ============================================================================

def load_historical_data():
    """Load historical detection and prediction data."""
    global historical_data
    if HISTORICAL_DATA_FILE.exists():
        try:
            with open(HISTORICAL_DATA_FILE, 'r') as f:
                historical_data = json.load(f)
        except:
            historical_data = []
    return historical_data

def save_historical_data():
    """Save historical data to file."""
    with open(HISTORICAL_DATA_FILE, 'w') as f:
        json.dump(historical_data, f, indent=2, default=str)

def load_unit_health():
    """Load unit health monitoring data."""
    global unit_health_data
    if UNIT_HEALTH_FILE.exists():
        try:
            with open(UNIT_HEALTH_FILE, 'r') as f:
                unit_health_data = json.load(f)
        except:
            unit_health_data = {}
    return unit_health_data

def save_unit_health():
    """Save unit health data to file."""
    with open(UNIT_HEALTH_FILE, 'w') as f:
        json.dump(unit_health_data, f, indent=2, default=str)

def initialize_unit_health():
    """Initialize unit health tracking for all units."""
    global unit_health_data
    if not unit_health_data:
        # Create default units
        for unit_id in range(1, 11):  # 10 units
            unit_health_data[f'unit_{unit_id}'] = {
                'unit_id': f'unit_{unit_id}',
                'location': {'lat': 7.0 + np.random.random() * 2, 'lon': 80.0 + np.random.random() * 2},
                'status': 'operational',
                'battery_level': np.random.uniform(0.7, 1.0),
                'sensor_health': np.random.uniform(0.8, 1.0),
                'last_maintenance': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                'detections_24h': np.random.randint(0, 15),
                'deterrent_uses_24h': np.random.randint(0, 10),
                'uptime_percentage': np.random.uniform(95, 100),
                'signal_strength': np.random.uniform(0.7, 1.0),
                'temperature': np.random.uniform(25, 35),
                'alerts': []
            }
        save_unit_health()

# ============================================================================
# TIME SERIES PREDICTION MODELS
# ============================================================================

class TimeSeriesPredictor:
    """Predicts future elephant activity patterns."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, historical_data, lookback=7):
        """Prepare features for time series prediction."""
        if len(historical_data) < lookback:
            return None, None
            
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.Timestamp.now()))
        df = df.sort_values('timestamp')
        
        # Create time-based features
        features = []
        targets = []
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            
            # Aggregate features from window
            feature_vec = [
                window['elephant_count'].mean() if 'elephant_count' in window.columns else 0,
                window['elephant_count'].max() if 'elephant_count' in window.columns else 0,
                window['aggression_level'].mean() if 'aggression_level' in window.columns else 0,
                window['proximity_to_boundary_m'].mean() if 'proximity_to_boundary_m' in window.columns else 0,
                window['hour_of_day'].mean() if 'hour_of_day' in window.columns else 12,
                window['is_night'].mean() if 'is_night' in window.columns else 0,
                window['temperature_celsius'].mean() if 'temperature_celsius' in window.columns else 28,
                window['weather_condition'].mode()[0] if 'weather_condition' in window.columns else 0,
            ]
            
            # Target: next period's elephant count
            target = df.iloc[i]['elephant_count'] if 'elephant_count' in df.iloc[i] else 0
            
            features.append(feature_vec)
            targets.append(target)
        
        if len(features) == 0:
            return None, None
            
        return np.array(features), np.array(targets)
    
    def train(self, historical_data):
        """Train the prediction model."""
        X, y = self.prepare_features(historical_data)
        
        if X is None or len(X) < 10:
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True
    
    def predict(self, historical_data, n_hours=24):
        """Predict elephant activity for next n_hours."""
        if not self.is_trained:
            return None
        
        X, _ = self.prepare_features(historical_data)
        if X is None or len(X) == 0:
            return None
        
        predictions = []
        current_features = X[-1].copy()
        
        for hour in range(n_hours):
            X_scaled = self.scaler.transform([current_features])
            pred = self.model.predict(X_scaled)[0]
            predictions.append({
                'hour': hour,
                'predicted_elephant_count': max(0, pred),
                'timestamp': (datetime.now() + timedelta(hours=hour)).isoformat()
            })
            
            # Update features for next prediction (simplified)
            current_features[0] = pred  # Update elephant count
            current_features[4] = (datetime.now().hour + hour) % 24  # Update hour
        
        return predictions

# ============================================================================
# DECISION SUPPORT SYSTEM
# ============================================================================

class DecisionSupportSystem:
    """Provides recommendations for wildlife officers."""
    
    def analyze_situation(self, current_data, historical_data, unit_health):
        """Analyze current situation and provide recommendations."""
        recommendations = []
        alerts = []
        priority = 'low'
        
        # Check elephant activity trends
        if len(historical_data) >= 7:
            recent_detections = [d.get('elephant_count', 0) for d in historical_data[-7:]]
            avg_recent = np.mean(recent_detections)
            avg_historical = np.mean([d.get('elephant_count', 0) for d in historical_data[-30:]]) if len(historical_data) >= 30 else avg_recent
            
            if avg_recent > avg_historical * 1.5:
                recommendations.append({
                    'type': 'activity_increase',
                    'priority': 'high',
                    'message': f'Significant increase in elephant activity detected. Recent average: {avg_recent:.1f} vs historical: {avg_historical:.1f}',
                    'action': 'Consider deploying additional units or increasing patrol frequency'
                })
                priority = 'high'
        
        # Check unit health
        critical_units = []
        for unit_id, health in unit_health.items():
            if health.get('battery_level', 1.0) < 0.3:
                critical_units.append(unit_id)
                alerts.append({
                    'type': 'low_battery',
                    'unit': unit_id,
                    'message': f'{unit_id} has low battery ({health["battery_level"]*100:.1f}%)',
                    'priority': 'high'
                })
                priority = 'high'
            
            if health.get('sensor_health', 1.0) < 0.7:
                alerts.append({
                    'type': 'sensor_degradation',
                    'unit': unit_id,
                    'message': f'{unit_id} sensor health degraded ({health["sensor_health"]*100:.1f}%)',
                    'priority': 'medium'
                })
                if priority == 'low':
                    priority = 'medium'
        
        if critical_units:
            recommendations.append({
                'type': 'unit_maintenance',
                'priority': 'high',
                'message': f'{len(critical_units)} unit(s) require immediate attention',
                'action': f'Schedule maintenance for: {", ".join(critical_units)}',
                'units': critical_units
            })
        
        # Check habituation risk
        if current_data:
            habituation_risk = current_data.get('habituation_risk', 0)
            if habituation_risk > 0.7:
                recommendations.append({
                    'type': 'habituation_warning',
                    'priority': 'medium',
                    'message': f'High habituation risk detected ({habituation_risk*100:.1f}%)',
                    'action': 'Consider rotating deterrent strategies or implementing cooldown period'
                })
                if priority == 'low':
                    priority = 'medium'
        
        # Check weather conditions
        if current_data and current_data.get('weather_condition', 0) == 2:
            recommendations.append({
                'type': 'weather_alert',
                'priority': 'medium',
                'message': 'Severe weather conditions may affect detection accuracy',
                'action': 'Monitor sensor readings closely and consider manual verification'
            })
        
        # Generate strategic recommendations
        if len(historical_data) >= 14:
            # Analyze patterns
            hourly_patterns = defaultdict(list)
            for entry in historical_data[-14:]:
                hour = entry.get('hour_of_day', 12)
                hourly_patterns[hour].append(entry.get('elephant_count', 0))
            
            peak_hours = sorted(hourly_patterns.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
            if peak_hours:
                recommendations.append({
                    'type': 'pattern_analysis',
                    'priority': 'low',
                    'message': f'Peak activity hours identified: {", ".join([f"{h[0]}:00" for h in peak_hours])}',
                    'action': 'Consider increasing monitoring during these hours'
                })
        
        return {
            'recommendations': recommendations,
            'alerts': alerts,
            'overall_priority': priority,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# ROUTE FUNCTIONS (to be registered in main app)
# ============================================================================

def register_monitoring_routes(app):
    """Register all monitoring routes with the Flask app."""
    
    @app.route('/monitoring')
    def monitoring_dashboard():
        """Serve the monitoring dashboard."""
        return render_template('monitoring.html')
    
    @app.route('/api/monitoring/status', methods=['GET'])
    def get_monitoring_status():
        """Get overall system status."""
        load_historical_data()
        load_unit_health()
        initialize_unit_health()
        
        total_detections = len(historical_data)
        active_units = sum(1 for h in unit_health_data.values() if h.get('status') == 'operational')
        total_units = len(unit_health_data)
        
        return jsonify({
            'success': True,
            'status': {
                'total_detections': total_detections,
                'active_units': active_units,
                'total_units': total_units,
                'system_health': active_units / total_units if total_units > 0 else 0,
                'last_update': datetime.now().isoformat()
            }
        })
    
    @app.route('/api/monitoring/detections', methods=['GET', 'POST'])
    def handle_detections():
        """Record or retrieve elephant detections."""
        global historical_data
        
        if request.method == 'POST':
            # Record new detection
            detection_data = request.json
            detection_data['timestamp'] = datetime.now().isoformat()
            detection_data['id'] = len(historical_data) + 1
            
            historical_data.append(detection_data)
            
            # Keep only last 1000 entries
            if len(historical_data) > 1000:
                historical_data = historical_data[-1000:]
            
            save_historical_data()
            
            return jsonify({
                'success': True,
                'message': 'Detection recorded',
                'detection_id': detection_data['id']
            })
        
        else:
            # Retrieve detections
            load_historical_data()
            
            # Filter by date range if provided
            days = request.args.get('days', type=int)
            if days:
                cutoff = datetime.now() - timedelta(days=days)
                filtered = [d for d in historical_data 
                           if datetime.fromisoformat(d.get('timestamp', datetime.now().isoformat())) > cutoff]
            else:
                filtered = historical_data
            
            return jsonify({
                'success': True,
                'detections': filtered[-100:],  # Return last 100
                'total': len(historical_data)
            })
    
    @app.route('/api/monitoring/units', methods=['GET'])
    def get_unit_health():
        """Get health status of all monitoring units."""
        load_unit_health()
        initialize_unit_health()
        
        return jsonify({
            'success': True,
            'units': list(unit_health_data.values())
        })
    
    @app.route('/api/monitoring/units/<unit_id>', methods=['GET', 'PUT'])
    def handle_unit(unit_id):
        """Get or update specific unit health."""
        load_unit_health()
        initialize_unit_health()
        
        if request.method == 'PUT':
            # Update unit health
            update_data = request.json
            if unit_id in unit_health_data:
                unit_health_data[unit_id].update(update_data)
                unit_health_data[unit_id]['last_update'] = datetime.now().isoformat()
                save_unit_health()
                
                return jsonify({
                    'success': True,
                    'message': f'Unit {unit_id} updated',
                    'unit': unit_health_data[unit_id]
                })
        
        # GET unit health
        if unit_id in unit_health_data:
            return jsonify({
                'success': True,
                'unit': unit_health_data[unit_id]
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Unit not found'
            }), 404
    
    @app.route('/api/monitoring/predictions', methods=['GET', 'POST'])
    def handle_predictions():
        """Generate or retrieve future predictions."""
        load_historical_data()
        
        if request.method == 'POST':
            # Generate new predictions
            n_hours = request.json.get('n_hours', 24)
            
            predictor = TimeSeriesPredictor()
            if len(historical_data) >= 10:
                predictor.train(historical_data)
                predictions = predictor.predict(historical_data, n_hours)
                
                if predictions:
                    return jsonify({
                        'success': True,
                        'predictions': predictions,
                        'model_trained': True
                    })
            
            # Fallback: generate synthetic predictions
            predictions = []
            base_count = np.mean([d.get('elephant_count', 0) for d in historical_data[-7:]]) if historical_data else 2.0
            for hour in range(n_hours):
                # Add some variation
                pred_count = max(0, base_count + np.random.normal(0, 0.5))
                predictions.append({
                    'hour': hour,
                    'predicted_elephant_count': round(pred_count, 1),
                    'timestamp': (datetime.now() + timedelta(hours=hour)).isoformat(),
                    'confidence': 0.7 if len(historical_data) >= 10 else 0.5
                })
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'model_trained': len(historical_data) >= 10
            })
        else:
            # Return cached predictions or generate new
            return handle_predictions()  # Generate on GET too
    
    @app.route('/api/monitoring/recommendations', methods=['GET'])
    def get_recommendations():
        """Get decision support recommendations."""
        load_historical_data()
        load_unit_health()
        initialize_unit_health()
        
        # Get most recent detection
        current_data = historical_data[-1] if historical_data else None
        
        dss = DecisionSupportSystem()
        analysis = dss.analyze_situation(current_data, historical_data, unit_health_data)
        
        return jsonify({
            'success': True,
            **analysis
        })
    
    @app.route('/api/monitoring/analytics', methods=['GET'])
    def get_analytics():
        """Get comprehensive analytics and insights."""
        load_historical_data()
        load_unit_health()
        
        if not historical_data:
            return jsonify({
                'success': True,
                'analytics': {
                    'total_detections': 0,
                    'message': 'Insufficient data for analytics'
                }
            })
        
        df = pd.DataFrame(historical_data)
        
        # Time-based analytics
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['date'] = df['timestamp'].dt.date
        
        analytics = {
            'overview': {
                'total_detections': len(historical_data),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                }
            },
            'elephant_activity': {
                'avg_count': float(df['elephant_count'].mean()) if 'elephant_count' in df.columns else 0,
                'max_count': int(df['elephant_count'].max()) if 'elephant_count' in df.columns else 0,
                'total_elephants': int(df['elephant_count'].sum()) if 'elephant_count' in df.columns else 0
            },
            'temporal_patterns': {}
        }
        
        # Hourly patterns
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour')['elephant_count'].mean() if 'elephant_count' in df.columns else pd.Series()
            analytics['temporal_patterns']['hourly'] = {
                int(h): float(v) for h, v in hourly_avg.items()
            }
        
        # Aggression patterns
        if 'aggression_level' in df.columns:
            aggression_dist = df['aggression_level'].value_counts().to_dict()
            analytics['aggression_distribution'] = {int(k): int(v) for k, v in aggression_dist.items()}
        
        # Unit performance
        unit_stats = {}
        for unit_id, health in unit_health_data.items():
            unit_stats[unit_id] = {
                'detections': health.get('detections_24h', 0),
                'deterrent_uses': health.get('deterrent_uses_24h', 0),
                'uptime': health.get('uptime_percentage', 0),
                'battery': health.get('battery_level', 0)
            }
        analytics['unit_performance'] = unit_stats
        
        # Effectiveness trends
        if 'deterrence_effectiveness' in df.columns:
            effectiveness_dist = df['deterrence_effectiveness'].value_counts().to_dict()
            analytics['effectiveness_distribution'] = {int(k): int(v) for k, v in effectiveness_dist.items()}
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
    
    @app.route('/api/monitoring/deterrence/predict', methods=['POST'])
    def predict_deterrence():
        """Use the existing deterrence model for predictions."""
        global deterrence_model
        
        if not deterrence_model.is_trained:
            # Initialize and train if needed
            deterrence_model.initialize(5000)
            deterrence_model.train(100)
        
        input_data = request.json
        prediction = deterrence_model.predict(input_data)
        
        # Record this as a detection
        detection = input_data.copy()
        detection['prediction'] = prediction
        detection['timestamp'] = datetime.now().isoformat()
        
        historical_data.append(detection)
        save_historical_data()
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
