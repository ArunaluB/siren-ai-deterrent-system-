#!/bin/bash

echo "🐘 Starting Elephant Deterrence System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment found..."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install setuptools first
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run the application
echo ""
echo "Starting Flask server..."
echo "Open http://localhost:5001 in your browser"
echo ""
python app.py
