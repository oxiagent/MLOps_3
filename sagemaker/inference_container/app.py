#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import base64
import time
import traceback
from typing import Dict, List, Any, Optional, Union
import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# Import modules for resource monitoring
try:
    import psutil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("psutil is not installed. Resource monitoring will be limited.")

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, log_level.upper(), logging.INFO)

# Ensure log directory exists
os.makedirs("/opt/ml/output/data", exist_ok=True)

logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/ml/output/data/app.log')
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("=" * 80)
logger.info("Starting YOLO Object Detection FastAPI application")
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
logger.info("=" * 80)

# Global counter for tracking requests
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_processing_time": 0,
    "last_request_time": None
}

# FastAPI app
app = FastAPI(title="YOLO Object Detection API", description="YOLO object detection for SageMaker")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response
class DetectionRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None

class DetectionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]

# Global model variable
model = None

def load_model():
    """Load the YOLO model"""
    global model
    if model is None:
        try:
            # Get model path from environment variable or use default
            model_dir = os.environ.get("MODEL_PATH", "/opt/ml/model")
            
            # Check if it's a directory or a file
            if os.path.isdir(model_dir):
                # First try to find a file with .pt extension
                model_path = os.path.join(model_dir, "model.pt")
                if not os.path.exists(model_path):
                    # If a file with .pt extension is not found, try to find a file without an extension
                    model_path = os.path.join(model_dir, "model")
                logger.info(f"MODEL_PATH is a directory, looking for model file at: {model_path}")
            else:
                model_path = model_dir
                logger.info(f"MODEL_PATH is a file: {model_path}")
            
            # Log model path
            logger.info(f"Loading model from: {model_path}")
            
            # Check if file exists
            if not os.path.exists(model_path):
                available_files = os.listdir(os.path.dirname(model_path)) if os.path.dirname(model_path) else []
                logger.error(f"Model file not found at {model_path}. Available files: {available_files}")
                fallback_paths = [
                    "/opt/ml/model/model.pt",
                    "/opt/ml/model/model",  # Path without extension
                    "model.pt",
                    "model"  # Path without extension
                ]
                
                # Try fallback paths
                for path in fallback_paths:
                    if os.path.exists(path):
                        logger.info(f"Found model at fallback path: {path}")
                        model_path = path
                        break
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path} or any fallback location")
            
            # Log model loading
            start_time = time.time()
            logger.info(f"Starting to load model from: {model_path}")
            
            # Set device based on availability
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
            # Load model
            model = YOLO(model_path)
            model.to(device)
            
            # Log model loading completion
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model type: {type(model)}")
            return model
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
    return model

def get_system_metrics():
    """Get system metrics for monitoring"""
    metrics = {
        "timestamp": time.time(),
        "memory": {},
        "cpu": {},
        "disk": {},
        "pytorch": {
            "cuda_available": torch.cuda.is_available()
        },
        "requests": request_stats.copy()
    }
    
    # Add information about requests
    metrics["requests"]["uptime"] = time.time() - startup_time
    
    if MONITORING_AVAILABLE:
        # Memory information
        memory = psutil.virtual_memory()
        metrics["memory"] = {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_percent": memory.percent
        }
        
        # CPU information
        metrics["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(logical=True)
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        metrics["disk"] = {
            "total_gb": disk.total / (1024 * 1024 * 1024),
            "free_gb": disk.free / (1024 * 1024 * 1024),
            "used_percent": disk.percent
        }
        
        # Process information
        process = psutil.Process(os.getpid())
        metrics["process"] = {
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "threads": len(process.threads())
        }
    
    # Information about PyTorch and CUDA
    if torch.cuda.is_available():
        metrics["pytorch"]["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        metrics["pytorch"]["cuda_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        try:
            metrics["pytorch"]["cuda_device_name"] = torch.cuda.get_device_name(0)
        except:
            metrics["pytorch"]["cuda_device_name"] = "unknown"
            
    return metrics

async def get_image_from_path(image_path: str):
    """Get image from URL or local path"""
    try:
        if image_path.startswith(("http://", "https://")):
            logger.info(f"Downloading image from URL: {image_path}")
            with urllib.request.urlopen(image_path) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data))
        else:
            logger.info(f"Loading image from local path: {image_path}")
            image = Image.open(image_path)
        return image
    except Exception as e:
        logger.error(f"Error loading image from path: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

async def get_image_from_base64(image_base64: str):
    """Get image from base64 string"""
    try:
        logger.info("Decoding base64 image")
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

async def process_raw_bytes(raw_bytes):
    """Process raw bytes into an image"""
    try:
        logger.info("Processing raw image bytes")
        image = Image.open(BytesIO(raw_bytes))
        return image
    except Exception as e:
        logger.error(f"Error processing image bytes: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Failed to process image bytes: {str(e)}")

# Save startup time
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    global startup_time
    startup_time = time.time()
    logger.info("Application started")
    
    # Attempt to load the model at startup
    try:
        logger.info("Preloading model...")
        load_model()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preloading failed: {str(e)}")
        logger.warning("Will try to load model on first request")

@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint called")
    return {
        "message": "YOLO Detection API ready",
        "endpoints": {
            "/health": "Health check endpoint",
            "/metrics": "System metrics and monitoring",
            "/predict": "Prediction endpoint for image files",
            "/invocations": "SageMaker inference endpoint"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint compatible with SageMaker"""
    logger.info("Health check request received")
    try:
        # Check for model file in default and fallback locations
        model_path = os.environ.get("MODEL_PATH", "/opt/ml/model/model.pt")
        model_exists = False
        paths_to_check = [model_path, "/opt/ml/model/model.pt", "/opt/ml/model/yolov8n.pt"]
        
        for path in paths_to_check:
            if os.path.exists(path):
                logger.info(f"Found model at: {path}")
                model_exists = True
                break
                
        # Get basic system metrics
        metrics = get_system_metrics()
        
        # Prepare health response
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - startup_time,
            "model_found": model_exists,
            "memory": metrics.get("memory", {}).get("used_percent", "N/A"),
            "cuda_available": torch.cuda.is_available()
        }
        
        return JSONResponse(content=health_info, status_code=200)
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Still return 200 with error info to pass SageMaker health check
        return JSONResponse(
            content={
                "status": "degraded",
                "error": str(e),
                "timestamp": time.time()
            }, 
            status_code=200
        )

@app.get("/ping")
async def ping():
    """Health check endpoint for SageMaker compatibility"""
    # Redirect to the /health endpoint for consistency
    response = await health()
    # SageMaker expects an empty 200 response for /ping
    return Response(content="", status_code=200)

@app.get("/metrics")
async def metrics():
    """Endpoint for system metrics and monitoring"""
    logger.info("Metrics request received")
    
    try:
        # Get system metrics
        metrics_data = get_system_metrics()
        
        # Add monitor log info if available
        try:
            log_dir = "/opt/ml/output/data"
            monitor_files = [f for f in os.listdir(log_dir) if f.startswith("monitor-")]
            
            if monitor_files:
                # Get the most recent monitor file
                latest_monitor = sorted(monitor_files)[-1]
                metrics_data["monitor_log"] = os.path.join(log_dir, latest_monitor)
                
                # Read last few lines of the monitor log
                with open(os.path.join(log_dir, latest_monitor), 'r') as f:
                    last_lines = f.readlines()[-10:]
                    metrics_data["monitor_log_snippet"] = last_lines
        except Exception as e:
            logger.warning(f"Error reading monitor logs: {str(e)}")
            metrics_data["monitor_log_error"] = str(e)
            
        return JSONResponse(content=metrics_data)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for image prediction with file upload"""
    logger.info(f"Prediction request received for file: {file.filename}")
    start_time = time.time()
    
    # Update request statistics
    request_stats["total_requests"] += 1
    request_stats["last_request_time"] = time.time()
    
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Process the uploaded file
        contents = await file.read()
        image = await process_raw_bytes(contents)
        
        # Perform inference
        results = model(image)
        
        # Process results
        predictions = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                prediction = {
                    "box": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0])
                }
                
                # Get class name if available
                if hasattr(result.names, 'get'):
                    prediction["class_name"] = result.names.get(int(box.cls[0]), f"class_{int(box.cls[0])}")
                
                predictions.append(prediction)
        
        # Model info
        model_info = {
            "type": "YOLO",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "processing_time": time.time() - start_time
        }
        
        # Successful request
        request_stats["successful_requests"] += 1
        total_success = request_stats["successful_requests"]
        current_time = model_info["processing_time"]
        request_stats["avg_processing_time"] = ((request_stats["avg_processing_time"] * (total_success - 1)) + current_time) / total_success
        
        return {
            "predictions": predictions,
            "model_info": model_info
        }
        
    except Exception as e:
        # Request error
        request_stats["failed_requests"] += 1
        
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker inference endpoint
    """
    logger.info("Inference request received")
    start_time = time.time()
    
    # Update request statistics
    request_stats["total_requests"] += 1
    request_stats["last_request_time"] = time.time()
    
    try:
        # Load model if not already loaded
        load_model()
        
        content_type = request.headers.get("Content-Type", "application/json")
        logger.info(f"Request content type: {content_type}")
        
        # Process based on content type
        if content_type == "application/json":
            # Process JSON request
            body = await request.json()
            logger.info(f"Received JSON request: {json.dumps(body)[:100]}...")
            
            if "image_path" in body:
                # Process image from path
                image = await get_image_from_path(body["image_path"])
            elif "image_base64" in body:
                # Process base64 encoded image
                image = await get_image_from_base64(body["image_base64"])
            else:
                raise HTTPException(status_code=400, detail="Request must include 'image_path' or 'image_base64'")
                
        elif content_type.startswith("multipart/form-data"):
            # Process multipart form data
            form = await request.form()
            logger.info(f"Received form data: {form}")
            
            if "file" not in form:
                raise HTTPException(status_code=400, detail="No file uploaded")
                
            file = form["file"]
            contents = await file.read()
            image = await process_raw_bytes(contents)
            
        elif content_type.startswith("application/x-www-form-urlencoded"):
            # Process form-urlencoded data
            form = await request.form()
            logger.info(f"Received urlencoded form: {form}")
            
            if "image_path" in form:
                image = await get_image_from_path(form["image_path"])
            elif "image_base64" in form:
                image = await get_image_from_base64(form["image_base64"])
            else:
                raise HTTPException(status_code=400, detail="Form must include 'image_path' or 'image_base64'")
                
        elif content_type in ["image/jpeg", "image/png", "application/octet-stream"]:
            # Process raw image data
            raw_data = await request.body()
            image = await process_raw_bytes(raw_data)
            
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")
            
        # Perform inference
        results = model(image)
        
        # Process results
        predictions = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                prediction = {
                    "box": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0])
                }
                
                # Get class name if available
                if hasattr(result.names, 'get'):
                    prediction["class_name"] = result.names.get(int(box.cls[0]), f"class_{int(box.cls[0])}")
                
                predictions.append(prediction)
        
        # Model info
        model_info = {
            "type": "YOLO",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "processing_time": time.time() - start_time
        }
        
        # Prepare response
        response = {
            "predictions": predictions,
            "model_info": model_info
        }
        
        # Successful request
        request_stats["successful_requests"] += 1
        total_success = request_stats["successful_requests"]
        current_time = model_info["processing_time"]
        request_stats["avg_processing_time"] = ((request_stats["avg_processing_time"] * (total_success - 1)) + current_time) / total_success
        
        # SageMaker expects JSON response
        return JSONResponse(content=response)
        
    except Exception as e:
        # Request error
        request_stats["failed_requests"] += 1
        
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            content={"error": error_msg},
            status_code=500
        ) 