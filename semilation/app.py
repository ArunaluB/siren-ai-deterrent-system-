"""
Flask Backend for Elephant Deterrence System
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from model import ElephantDeterrenceModel
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Initialize model
model = ElephantDeterrenceModel()

# Global state
model_initialized = False
model_trained = False

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_model():
    """Initialize the model with dataset."""
    global model_initialized
    
    try:
        data = request.json
        n_samples = data.get('n_samples', 5000)
        
        result = model.initialize(n_samples)
        model_initialized = True
        
        return jsonify({
            'success': True,
            'message': 'Model initialized successfully',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the RL model with progress updates."""
    global model_trained
    
    if not model_initialized:
        return jsonify({
            'success': False,
            'message': 'Model not initialized'
        }), 400
    
    try:
        data = request.json
        n_episodes = data.get('n_episodes', 100)
        
        # Train model
        training_results = model.train(n_episodes)
        model_trained = True
        
        # Get final stats
        stats = model.get_model_stats()
        
        return jsonify({
            'success': True,
            'message': 'Training completed',
            'data': training_results,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction with the model."""
    if not model_trained:
        return jsonify({
            'success': False,
            'message': 'Model not trained yet'
        }), 400
    
    try:
        input_data = request.json
        
        # Validate required fields
        required_fields = [
            'hour_of_day', 'is_night', 'ambient_noise_db', 'weather_condition',
            'wind_speed_kmh', 'temperature_celsius', 'elephant_count', 'aggression_level',
            'proximity_to_boundary_m', 'movement_speed', 'deterrent_uses_24h',
            'days_since_last_use', 'cumulative_exposure_score', 'effectiveness_decay_factor',
            'human_proximity_m', 'crop_value_zone', 'boundary_segment_risk',
            'sensor_confidence', 'battery_level'
        ]
        
        # Check if all fields present
        for field in required_fields:
            if field not in input_data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400
        
        prediction = model.predict(input_data)
        
        # Update habituation engine
        model.habituation_engine.update(
            action_taken=prediction['action'],
            effectiveness=prediction['predicted_class'],
            timestamp=len(model.habituation_engine.history)
        )
        
        habituation_score = model.habituation_engine.get_habituation_score()
        cooldown_recommended = model.habituation_engine.recommend_cooldown()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'habituation': {
                'score': float(habituation_score),
                'cooldown_recommended': cooldown_recommended
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/generate-sample', methods=['GET'])
def generate_sample():
    """Generate a random sample data point."""
    try:
        sample = model.generate_sample_data()
        return jsonify({
            'success': True,
            'data': sample
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status."""
    return jsonify({
        'initialized': model_initialized,
        'trained': model_trained,
        'habituation_score': float(model.habituation_engine.get_habituation_score()) if model.habituation_engine.history else 0.0
    })

@app.route('/api/sound/<action>', methods=['POST'])
def trigger_sound(action):
    """Trigger sound playback for specific action."""
    import os
    from pathlib import Path
    
    # Map actions to sound files
    sound_mapping = {
        '0': None,  # OBSERVE - no sound
        '1': 'bees-82424.mp3',  # SPARSE_BIO - bee sound
        '2': 'tiger-light-roar-t-293716.mp3',  # DIRECTIONAL - light tiger roar
        '3': 'tiger-roar-104166.mp3',  # MULTI_SPECTRAL - tiger roar
        '4': 'tiger-roar-loudly-193229.mp3'  # HUMAN_ALERT - loud tiger roar
    }
    
    action_names = {
        '0': 'OBSERVE',
        '1': 'SPARSE_BIO',
        '2': 'DIRECTIONAL',
        '3': 'MULTI_SPECTRAL',
        '4': 'HUMAN_ALERT'
    }
    
    sound_file = sound_mapping.get(str(action))
    action_name = action_names.get(str(action), 'UNKNOWN')
    
    if sound_file:
        sound_path = Path('static/sounds') / sound_file
        if sound_path.exists():
            return jsonify({
                'success': True,
                'message': f'Sound triggered: {action_name}',
                'sound': sound_file,
                'sound_path': f'/static/sounds/{sound_file}',
                'action': int(action),
                'action_name': action_name
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Sound file not found: {sound_file}',
                'sound': None,
                'action': int(action),
                'action_name': action_name
            }), 404
    else:
        return jsonify({
            'success': True,
            'message': f'No sound for action: {action_name}',
            'sound': None,
            'action': int(action),
            'action_name': action_name
        })

@app.route('/api/sounds/list', methods=['GET'])
def list_sounds():
    """List all available sound files."""
    import os
    from pathlib import Path
    
    sounds_dir = Path('static/sounds')
    sound_files = []
    
    if sounds_dir.exists():
        for file in sounds_dir.iterdir():
            if file.suffix.lower() in ['.mp3', '.wav', '.ogg']:
                sound_files.append({
                    'filename': file.name,
                    'path': f'/static/sounds/{file.name}',
                    'size': file.stat().st_size
                })
    
    return jsonify({
        'success': True,
        'sounds': sound_files,
        'count': len(sound_files)
    })

@app.route('/api/stats', methods=['GET'])
def get_model_stats():
    """Get comprehensive model statistics."""
    if not model_trained:
        return jsonify({
            'success': False,
            'message': 'Model not trained yet'
        }), 400
    
    try:
        stats = model.get_model_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_scenarios():
    """Run simulation with multiple scenarios and update monitoring system."""
    if not model_trained:
        return jsonify({
            'success': False,
            'message': 'Model not trained yet'
        }), 400
    
    try:
        data = request.json
        n_steps = data.get('n_steps', 10)
        
        scenarios = model.simulate_scenario(n_steps)
        
        # Send simulation data to monitoring system
        try:
            from monitoring_backend import (
                historical_data, unit_health_data, 
                save_historical_data, save_unit_health,
                load_historical_data, load_unit_health
            )
            from datetime import datetime
            import random
            
            # Load current data
            load_historical_data()
            load_unit_health()
            
            # Record each scenario as a detection
            for scenario in scenarios:
                detection = scenario['input'].copy()
                detection['prediction'] = scenario['prediction']
                detection['habituation_score'] = scenario.get('habituation_score', 0)
                detection['cooldown_recommended'] = scenario.get('cooldown_recommended', False)
                detection['timestamp'] = datetime.now().isoformat()
                detection['id'] = len(historical_data) + 1
                detection['source'] = 'simulation'
                
                historical_data.append(detection)
            
            # Keep only last 1000 entries
            if len(historical_data) > 1000:
                historical_data[:] = historical_data[-1000:]
            
            save_historical_data()
            
            # Update unit health based on simulation activity
            # Randomly assign detections to units and update their stats
            for unit_id in unit_health_data.keys():
                unit = unit_health_data[unit_id]
                # Simulate some activity
                unit['detections_24h'] = min(unit.get('detections_24h', 0) + random.randint(0, 2), 50)
                unit['deterrent_uses_24h'] = min(unit.get('deterrent_uses_24h', 0) + random.randint(0, 1), 20)
                # Slight battery drain
                unit['battery_level'] = max(0, unit.get('battery_level', 1.0) - random.uniform(0, 0.01))
                unit['last_update'] = datetime.now().isoformat()
            
            save_unit_health()
            
        except Exception as e:
            print(f"Warning: Could not update monitoring system: {e}")
        
        return jsonify({
            'success': True,
            'scenarios': scenarios,
            'n_steps': n_steps,
            'monitoring_updated': True
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Import and register monitoring routes
try:
    from monitoring_backend import register_monitoring_routes
    register_monitoring_routes(app)
except ImportError as e:
    print(f"Warning: Monitoring backend not available: {e}")
except Exception as e:
    print(f"Warning: Error loading monitoring routes: {e}")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/sounds', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Use port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
