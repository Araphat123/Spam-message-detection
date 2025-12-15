#!/bin/bash

# Navigate to the project root (assuming script is in project root)
cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install pandas scikit-learn flask
else
    source venv/bin/activate
fi

# Run the app
echo "Starting Spam Detection App..."
cd spam_app
python3 app.py
