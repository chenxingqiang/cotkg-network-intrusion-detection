#!/bin/bash

# Environment management script for CoTKG-IDS

function setup_env() {
    echo "Setting up Python virtual environment..."
    python -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo "Environment setup complete!"
}

function clean_env() {
    echo "Cleaning environment..."
    deactivate 2>/dev/null
    rm -rf venv
    rm -rf *.egg-info
    rm -rf build dist
    find . -type d -name "__pycache__" -exec rm -rf {} +
    echo "Environment cleaned!"
}

case "$1" in
    "setup")
        setup_env
        ;;
    "clean")
        clean_env
        ;;
    *)
        echo "Usage: $0 {setup|clean}"
        exit 1
        ;;
esac
