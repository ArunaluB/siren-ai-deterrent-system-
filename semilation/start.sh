#!/bin/bash

echo "🐘 Starting Elephant Deterrence System..."
echo ""

# Kill any process on port 5000 (macOS AirPlay Receiver)
echo "Checking for processes on port 5000..."
PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "Found process $PID on port 5000. Attempting to stop it..."
    kill -9 $PID 2>/dev/null && echo "✓ Stopped process on port 5000" || echo "⚠ Could not stop process (may need admin rights)"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install setuptools first
echo "Checking dependencies..."
pip install --quiet --upgrade pip setuptools wheel 2>/dev/null
pip install --quiet -r requirements.txt 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Run the application on port 5001
echo ""
echo "🚀 Starting Flask server on port 5001..."
echo "📱 Open http://localhost:5001 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
