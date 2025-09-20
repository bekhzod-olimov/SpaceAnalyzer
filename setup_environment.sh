#!/bin/bash

# This script creates a new Python virtual environment and installs all the
# necessary dependencies with compatible versions to run the demo_dino.py script.

# --- Configuration ---
VENV_NAME="/home/bekhzod/envs/space_dino"
PYTHON_EXECUTABLE="python3.11" # Change if you use a different python version, e.g., python3

# --- Script Start ---
echo "--- Starting Environment Setup for Space Analyzer ---"

# Check if the chosen python version exists
if ! command -v $PYTHON_EXECUTABLE &> /dev/null
then
    echo "Error: Python executable '$PYTHON_EXECUTABLE' not found."
    echo "Please edit this script and set PYTHON_EXECUTABLE to your installed python3 command."
    exit 1
fi

# 1. Create a new virtual environment
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
else
    echo "Creating virtual environment: $VENV_NAME..."
    $PYTHON_EXECUTABLE -m venv $VENV_NAME
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# 3. Install PyTorch with a specific CUDA version (Critical Step)
echo "Installing PyTorch (this may take a few minutes)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 4. Install all other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Completion ---
echo ""
echo "--- Setup Complete! ---"
echo "To activate this environment in the future, run:"
echo "source $VENV_NAME/bin/activate"
echo ""
echo "Remember to set your CUDA environment variables before running the script:"
echo "export CUDA_HOME=/usr/local/cuda-12.6"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo ""
