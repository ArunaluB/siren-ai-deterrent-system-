# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Server
```bash
python app.py
```

### Step 3: Open Browser
Navigate to: `http://localhost:5000`

## 📋 Usage Flow

1. **Click "Initialize Model"** - Creates 5,000 sample dataset
2. **Click "Train Model"** - Trains RL agent (takes ~30 seconds)
3. **Click "Generate Sample Data"** - Fills input fields automatically
4. **Click "Run Prediction"** - See model decision and action
5. **Click "Play Deterrent Sound"** - Hear audio feedback

## 🎮 Advanced Features

### Real-Time Simulation
- Click **"Run Simulation (10 Steps)"** to see model behavior over multiple scenarios
- Watch habituation score change over time
- See cooldown recommendations

### Auto Simulation
- Click **"🔄 Auto Simulate"** for continuous automatic predictions
- Runs every 3 seconds with new random data
- Click again to stop

### View Statistics
After training, you'll see:
- Test accuracy, precision, recall, F1-score
- Top 10 most important features
- Training metrics charts

## 🎵 Sound Files (Optional)

Add MP3 files to `static/sounds/`:
- `observe.mp3`
- `sparse_bio.mp3`
- `directional.mp3`
- `multi_spectral.mp3`
- `human_alert.mp3`

If files are missing, the system uses Web Audio API as fallback.

## 📊 What You'll See

### Dashboard Sections:
1. **System Control** - Initialize, train, generate data
2. **Input Parameters** - 19 configurable inputs
3. **Action Selection** - Visual display of 5 actions
4. **Prediction Results** - Detailed output
5. **Metrics Dashboard** - Key performance indicators
6. **Q-Values Chart** - Action confidence visualization
7. **Training Metrics** - Accuracy and loss over time
8. **Feature Importance** - Top features ranked
9. **Sound Control** - Audio playback
10. **Model Statistics** - Comprehensive metrics
11. **Real-Time Simulation** - Multi-step scenarios

## 🐛 Troubleshooting

**Model not training?**
- Make sure you clicked "Initialize Model" first

**No sound playing?**
- Check browser console for errors
- System will use Web Audio API if MP3 files missing

**Charts not showing?**
- Wait for training to complete
- Refresh page if needed

## 💡 Tips

- Use "Generate Sample Data" to quickly test different scenarios
- Watch the habituation score - it increases with repeated deterrent use
- High habituation score triggers cooldown recommendations
- Auto simulation is great for demonstrating system behavior
