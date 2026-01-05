"""
Enhanced Elephant Deterrence System with RL & Habituation Prevention
=====================================================================
Complete model implementation with all features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import json
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_synthetic_dataset(n_samples=5000):
    """Generate realistic synthetic data for elephant deterrence scenarios."""
    
    data = {
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'is_night': np.random.binomial(1, 0.6, n_samples),
        'ambient_noise_db': np.random.normal(45, 10, n_samples),
        'weather_condition': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'wind_speed_kmh': np.random.exponential(10, n_samples),
        'temperature_celsius': np.random.normal(28, 5, n_samples),
        'elephant_count': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples, 
                                          p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01]),
        'aggression_level': np.random.choice([0, 1, 2, 3], n_samples, 
                                            p=[0.5, 0.3, 0.15, 0.05]),
        'proximity_to_boundary_m': np.random.exponential(50, n_samples),
        'movement_speed': np.random.choice([0, 1, 2], n_samples, 
                                          p=[0.4, 0.4, 0.2]),
        'deterrent_uses_24h': np.random.poisson(2, n_samples),
        'days_since_last_use': np.random.exponential(3, n_samples),
        'cumulative_exposure_score': np.random.gamma(2, 2, n_samples),
        'effectiveness_decay_factor': np.random.beta(5, 2, n_samples),
        'human_proximity_m': np.random.exponential(100, n_samples),
        'crop_value_zone': np.random.choice([0, 1, 2], n_samples, 
                                           p=[0.3, 0.5, 0.2]),
        'boundary_segment_risk': np.random.choice([0, 1, 2], n_samples, 
                                                  p=[0.4, 0.4, 0.2]),
        'sensor_confidence': np.random.beta(8, 2, n_samples),
        'battery_level': np.random.beta(7, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['ambient_noise_db'] = df['ambient_noise_db'].clip(30, 80)
    df['wind_speed_kmh'] = df['wind_speed_kmh'].clip(0, 50)
    df['temperature_celsius'] = df['temperature_celsius'].clip(15, 40)
    df['proximity_to_boundary_m'] = df['proximity_to_boundary_m'].clip(5, 200)
    df['human_proximity_m'] = df['human_proximity_m'].clip(20, 500)
    df['days_since_last_use'] = df['days_since_last_use'].clip(0, 20)
    df['cumulative_exposure_score'] = df['cumulative_exposure_score'].clip(0, 15)
    
    df['deterrence_effectiveness'] = generate_effectiveness_labels(df)
    
    return df

def generate_effectiveness_labels(df):
    """Generate deterrence effectiveness labels based on domain knowledge."""
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        success_prob = 0.5
        
        habituation_penalty = df.loc[i, 'cumulative_exposure_score'] / 15.0
        success_prob -= habituation_penalty * 0.3
        
        if df.loc[i, 'deterrent_uses_24h'] > 3:
            success_prob -= 0.2
        
        aggression_penalty = df.loc[i, 'aggression_level'] * 0.15
        success_prob -= aggression_penalty
        
        if df.loc[i, 'elephant_count'] > 4:
            success_prob -= 0.15
        
        if df.loc[i, 'weather_condition'] == 2:
            success_prob -= 0.2
        if df.loc[i, 'wind_speed_kmh'] > 30:
            success_prob -= 0.1
        
        if df.loc[i, 'ambient_noise_db'] > 60:
            success_prob -= 0.15
        
        if df.loc[i, 'sensor_confidence'] < 0.5:
            success_prob -= 0.1
        
        success_prob *= df.loc[i, 'effectiveness_decay_factor']
        success_prob = np.clip(success_prob, 0, 0.9)
        
        outcome = np.random.random()
        if outcome < success_prob * 0.7:
            labels[i] = 2
        elif outcome < success_prob:
            labels[i] = 1
        else:
            labels[i] = 0
    
    return labels

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create advanced features for RL decision-making."""
    df_eng = df.copy()
    
    df_eng['habituation_risk'] = (
        0.4 * (df_eng['cumulative_exposure_score'] / 15.0) +
        0.3 * (df_eng['deterrent_uses_24h'] / 10.0) +
        0.3 * (1 - df_eng['effectiveness_decay_factor'])
    )
    
    df_eng['env_difficulty'] = (
        0.3 * (df_eng['wind_speed_kmh'] / 50.0) +
        0.3 * (df_eng['ambient_noise_db'] / 80.0) +
        0.4 * (df_eng['weather_condition'] / 2.0)
    )
    
    df_eng['threat_level'] = (
        0.3 * (1 - df_eng['proximity_to_boundary_m'] / 200.0) +
        0.3 * (df_eng['aggression_level'] / 3.0) +
        0.2 * (df_eng['elephant_count'] / 8.0) +
        0.2 * (df_eng['movement_speed'] / 2.0)
    )
    
    df_eng['human_risk'] = (
        0.5 * (1 - df_eng['human_proximity_m'] / 500.0) +
        0.3 * (df_eng['boundary_segment_risk'] / 2.0) +
        0.2 * (df_eng['crop_value_zone'] / 2.0)
    )
    
    df_eng['system_reliability'] = (
        0.6 * df_eng['sensor_confidence'] +
        0.4 * df_eng['battery_level']
    )
    
    df_eng['optimal_time_window'] = (
        (df_eng['hour_of_day'] >= 18) | (df_eng['hour_of_day'] <= 6)
    ).astype(int)
    
    df_eng['critical_scenario'] = (
        (df_eng['threat_level'] > 0.6) & (df_eng['habituation_risk'] > 0.5)
    ).astype(int)
    
    df_eng['deterrent_freshness'] = np.exp(-df_eng['deterrent_uses_24h'] / 3.0)
    
    return df_eng

# ============================================================================
# RL AGENT
# ============================================================================

class AdaptiveDeterrenceRLAgent:
    """SARSA-based RL agent for adaptive elephant deterrence."""
    
    def __init__(self, n_features, n_actions=5, learning_rate=0.01, 
                 gamma=0.95, epsilon=0.2):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.weights = np.zeros((n_features, n_actions))
        self.eligibility = np.zeros((n_features, n_actions))
        self.lambda_trace = 0.9
        
    def get_features(self, state):
        return state
    
    def get_q_value(self, state, action):
        features = self.get_features(state)
        return np.dot(features, self.weights[:, action])
    
    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, next_action):
        features = self.get_features(state)
        
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        td_error = reward + self.gamma * next_q - current_q
        
        self.eligibility *= self.gamma * self.lambda_trace
        self.eligibility[:, action] += features
        
        self.weights += self.lr * td_error * self.eligibility
        
        return td_error
    
    def compute_reward(self, effectiveness, habituation_risk, human_risk, 
                       action_taken, aggression_level):
        reward = 0.0
        
        effectiveness_rewards = {0: -5, 1: 3, 2: 10}
        reward += effectiveness_rewards[effectiveness]
        
        habituation_penalty = -3.0 * habituation_risk
        reward += habituation_penalty
        
        if human_risk > 0.7 and action_taken == 4:
            reward += 5.0
        
        action_costs = [0, -0.5, -1.0, -1.5, -2.0]
        if human_risk < 0.3:
            reward += action_costs[action_taken]
        
        if aggression_level > 2 and action_taken in [2, 3]:
            reward -= 3.0
        
        return reward

# ============================================================================
# HABITUATION ENGINE
# ============================================================================

class HabituationEngine:
    """Tracks and predicts deterrent habituation over time."""
    
    def __init__(self, decay_rate=0.05):
        self.decay_rate = decay_rate
        self.history = []
        self.effectiveness_window = []
        
    def update(self, action_taken, effectiveness, timestamp):
        self.history.append({
            'timestamp': timestamp,
            'action': action_taken,
            'effectiveness': effectiveness
        })
        
        self.effectiveness_window.append(effectiveness)
        if len(self.effectiveness_window) > 20:
            self.effectiveness_window.pop(0)
    
    def get_habituation_score(self):
        if len(self.effectiveness_window) < 5:
            return 0.0
        
        recent = self.effectiveness_window[-10:]
        if len(recent) < 5:
            return 0.0
        
        x = np.arange(len(recent))
        y = np.array(recent)
        if len(x) > 1 and np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
            habituation = -slope
            return np.clip(habituation, 0, 1)
        return 0.0
    
    def recommend_cooldown(self):
        if len(self.history) < 3:
            return False
        
        recent_actions = [h['action'] for h in self.history[-5:]]
        if recent_actions.count(recent_actions[-1]) >= 3:
            return True
        
        if self.get_habituation_score() > 0.6:
            return True
        
        return False

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_rl_agent(agent, X_train, y_train, X_val, y_val, df_engineered, n_episodes=100):
    """Train RL agent using episodic SARSA."""
    train_accuracies = []
    val_accuracies = []
    train_rewards = []
    losses = []
    
    for episode in range(n_episodes):
        episode_reward = 0
        episode_loss = 0
        predictions = []
        
        indices = np.random.permutation(len(X_train))
        
        for i, idx in enumerate(indices):
            state = X_train[idx]
            true_effectiveness = y_train.iloc[idx]
            
            original_idx = y_train.index[idx]
            habituation_risk = df_engineered.loc[original_idx, 'habituation_risk']
            human_risk = df_engineered.loc[original_idx, 'human_risk']
            aggression = df_engineered.loc[original_idx, 'aggression_level']
            
            action = agent.get_action(state, training=True)
            predictions.append(action % 3)
            
            if i < len(indices) - 1:
                next_idx = indices[i + 1]
                next_state = X_train[next_idx]
                next_action = agent.get_action(next_state, training=True)
            else:
                next_state = state
                next_action = action
            
            reward = agent.compute_reward(
                effectiveness=true_effectiveness,
                habituation_risk=habituation_risk,
                human_risk=human_risk,
                action_taken=action,
                aggression_level=aggression
            )
            
            td_error = agent.update(state, action, reward, next_state, next_action)
            
            episode_reward += reward
            episode_loss += abs(td_error)
        
        train_acc = accuracy_score(y_train, predictions)
        train_accuracies.append(train_acc)
        train_rewards.append(episode_reward / len(X_train))
        losses.append(episode_loss / len(X_train))
        
        val_predictions = []
        for state in X_val:
            action = agent.get_action(state, training=False)
            val_predictions.append(action % 3)
        val_acc = accuracy_score(y_val, val_predictions)
        val_accuracies.append(val_acc)
    
    return train_accuracies, val_accuracies, train_rewards, losses

# ============================================================================
# MODEL INITIALIZATION & LOADING
# ============================================================================

class ElephantDeterrenceModel:
    """Main model class for prediction and training."""
    
    def __init__(self):
        self.agent = None
        self.scaler = StandardScaler()
        self.df = None
        self.df_engineered = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.habituation_engine = HabituationEngine()
        self.is_trained = False
        
    def initialize(self, n_samples=5000):
        """Initialize and prepare the model."""
        self.df = generate_synthetic_dataset(n_samples)
        self.df_engineered = engineer_features(self.df)
        
        X = self.df_engineered.drop('deterrence_effectiveness', axis=1)
        y = self.df_engineered['deterrence_effectiveness']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
        
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.agent = AdaptiveDeterrenceRLAgent(
            n_features=self.X_train.shape[1],
            n_actions=5,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=0.2
        )
        
        return {
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'features': self.X_train.shape[1]
        }
    
    def train(self, n_episodes=100):
        """Train the RL agent."""
        train_acc, val_acc, rewards, losses = train_rl_agent(
            self.agent, self.X_train, self.y_train, 
            self.X_val, self.y_val, self.df_engineered, n_episodes
        )
        self.is_trained = True
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'rewards': rewards,
            'losses': losses
        }
    
    def predict(self, input_data):
        """Predict action and effectiveness for given input."""
        if not self.is_trained:
            return None
        
        # Convert input to DataFrame if dict
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
            df_eng = engineer_features(df_input)
            X_input = df_eng.drop('deterrence_effectiveness', axis=1, errors='ignore')
        else:
            X_input = input_data
            # If not dict, we need to engineer features - but this case shouldn't happen
            # For safety, create a dummy df_eng
            df_eng = pd.DataFrame({
                'habituation_risk': [0.5],
                'human_risk': [0.5],
                'threat_level': [0.5]
            })
        
        # Scale features
        X_scaled = self.scaler.transform(X_input)
        
        # Get action
        state = X_scaled[0]
        action = self.agent.get_action(state, training=False)
        predicted_class = action % 3
        
        # Get Q-values
        q_values = [self.agent.get_q_value(state, a) for a in range(self.agent.n_actions)]
        
        # Get additional metrics
        if isinstance(input_data, dict):
            habituation_risk = df_eng['habituation_risk'].iloc[0]
            human_risk = df_eng['human_risk'].iloc[0]
            threat_level = df_eng['threat_level'].iloc[0]
        else:
            habituation_risk = 0.5
            human_risk = 0.5
            threat_level = 0.5
        
        return {
            'action': int(action),
            'action_name': ['OBSERVE', 'SPARSE_BIO', 'DIRECTIONAL', 'MULTI_SPECTRAL', 'HUMAN_ALERT'][action],
            'predicted_class': int(predicted_class),
            'class_name': ['Failed', 'Partial', 'Success'][predicted_class],
            'q_values': [float(q) for q in q_values],
            'habituation_risk': float(habituation_risk),
            'human_risk': float(human_risk),
            'threat_level': float(threat_level),
            'confidence': float(max(q_values) - min(q_values)) if len(q_values) > 0 else 0.0
        }
    
    def generate_sample_data(self):
        """Generate a single sample data point."""
        if self.df is None:
            # Generate a minimal dataset if not initialized
            sample = generate_synthetic_dataset(1)
        else:
            # Use the same distribution as training data
            sample = generate_synthetic_dataset(1)
        sample_dict = sample.iloc[0].to_dict()
        # Remove the target variable if present
        if 'deterrence_effectiveness' in sample_dict:
            del sample_dict['deterrence_effectiveness']
        return sample_dict
    
    def get_model_stats(self):
        """Get comprehensive model statistics."""
        if not self.is_trained:
            return None
        
        # Get test predictions
        test_predictions = []
        for state in self.X_test:
            action = self.agent.get_action(state, training=False)
            test_predictions.append(action % 3)
        
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        test_precision = precision_score(self.y_test, test_predictions, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, test_predictions, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, test_predictions, average='weighted', zero_division=0)
        
        # Feature importance
        feature_importance = np.abs(self.agent.weights).mean(axis=1)
        feature_names = self.df_engineered.drop('deterrence_effectiveness', axis=1).columns
        
        top_features = []
        sorted_idx = np.argsort(feature_importance)[-10:]
        for idx in sorted_idx[::-1]:
            top_features.append({
                'name': feature_names[idx],
                'importance': float(feature_importance[idx])
            })
        
        return {
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'top_features': top_features,
            'n_features': int(self.X_train.shape[1]),
            'n_actions': int(self.agent.n_actions),
            'habituation_score': float(self.habituation_engine.get_habituation_score())
        }
    
    def simulate_scenario(self, n_steps=10):
        """Simulate multiple scenarios for demonstration."""
        scenarios = []
        for i in range(n_steps):
            sample = self.generate_sample_data()
            prediction = self.predict(sample)
            
            # Update habituation
            self.habituation_engine.update(
                action_taken=prediction['action'],
                effectiveness=prediction['predicted_class'],
                timestamp=len(self.habituation_engine.history)
            )
            
            scenarios.append({
                'step': i + 1,
                'input': sample,
                'prediction': prediction,
                'habituation_score': float(self.habituation_engine.get_habituation_score()),
                'cooldown_recommended': self.habituation_engine.recommend_cooldown()
            })
        
        return scenarios