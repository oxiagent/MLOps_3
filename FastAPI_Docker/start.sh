#!/bin/sh

# Перевіряємо, чи є WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY environment variable is not set"
    echo "Please run with: docker run -e WANDB_API_KEY=your_key ..."
    exit 1
fi

# Перевіряємо, чи завантажена вже модель
if [ ! -f "./model/linear_regression_model.pth" ]; then
    echo "Downloading model from WANDB..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download model"
        exit 1
    fi
    echo "Model downloaded successfully"
else
    echo "Model already exists, skipping download"
fi

# Запускаємо FastAPI сервер
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8080