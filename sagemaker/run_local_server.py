#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import uvicorn
import wandb
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import traceback

# Import utility functions
from utils import download_model_from_wandb, find_model_file

# Load environment variables
load_dotenv()

# Global variable to store the model
MODEL = None
app = FastAPI(title="YOLO Model API", description="API for YOLO model inference")

class Prediction(BaseModel):
    """Model for prediction result"""
    box: List[float]
    confidence: float
    class_id: int
    class_name: str

@app.get("/")
def read_root():
    """Root endpoint with information"""
    return {
        "message": "YOLO Model API Ready",
        "status": "active",
        "endpoints": {
            "/": "This page",
            "/predict": "POST endpoint for predictions",
            "/invocations": "SageMaker-compatible endpoint",
            "/docs": "FastAPI automatic documentation"
        },
        "example": {
            "request": {
                "method": "POST",
                "url": "/invocations",
                "body": "Binary image data"
            }
        }
    }

@app.post("/invocations", response_model=List[Prediction])
async def invocations(request: Request):
    """SageMaker-compatible endpoint for inference"""
    global MODEL
    
    if MODEL is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded"}
        )
    
    try:
        # Read raw binary data
        image_data = await request.body()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image format"}
            )
        
        # Make prediction
        results = MODEL(image)
        
        # Convert results to required format
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                pred = Prediction(
                    box=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=result.names[class_id]
                )
                predictions.append(pred.dict())
        
        return predictions
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(error_trace)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error during prediction: {str(e)}"}
        )

@app.post("/predict", response_model=List[Prediction])
async def predict(request: Request):
    """Endpoint for making predictions - alias for /invocations"""
    return await invocations(request)

def load_yolo_model(model_path):
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully")
        return model
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully")
        return model

def main():
    parser = argparse.ArgumentParser(description="Start FastAPI server for model inference")
    parser.add_argument("--artifact", type=str, 
              default="s-oksana-set-university/linear-regression-pytorch/linear_regression_model",
                     help="Artifact path in format 'entity/project/artifact_name'")
    parser.add_argument("--version", type=str, default="v0",
                     help="Artifact version (default: v0)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Server port (default: 8000)")
    
    args = parser.parse_args()
    
    # Check if WANDB_API_KEY is set
    if not os.getenv("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY not set in environment variables")
        sys.exit(1)
    
    # Initialize wandb
    try:
        wandb.init(project="linear-regression-pytorch", entity="s-oksana-set-university", job_type="inference")
    except Exception as e:
        print(f"Error initializing wandb: {str(e)}")
        sys.exit(1)
    
    # Download model from Weights & Biases
    try:
        print(f"Downloading model artifact: {args.artifact}:{args.version}")
        artifact = wandb.use_artifact(f"{args.artifact}:{args.version}")
        model_dir = artifact.download()
        print(f"Model downloaded to: {model_dir}")
        
        # Find model file in the downloaded directory
        model_path = find_model_file(model_dir)
        if not model_path:
            supported_extensions = [".pt", ".pth", ".onnx", ".pb"]
            print(f"Error: No model file found in {model_dir}")
            print(f"Supported extensions: {supported_extensions}")
            print(f"Directory contents: {os.listdir(model_dir)}")
            sys.exit(1)
        
        print(f"Found model at: {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)
    
    # Load model into global variable
    global MODEL
    MODEL = load_yolo_model(model_path)
    
    # Start FastAPI server
    print(f"Starting FastAPI server on http://{args.host}:{args.port}")
    print(f"SageMaker-compatible endpoint: http://{args.host}:{args.port}/invocations")
    print(f"View documentation at http://{args.host}:{args.port}/docs")
    print(f"Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 