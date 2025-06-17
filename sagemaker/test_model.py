#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import requests
import numpy as np
import cv2
import colorsys
import boto3
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import base64
from utils import get_aws_credentials

# Load environment variables
load_dotenv()

def visualize_predictions(image_path: str, predictions: List[Dict], save_path: str = None) -> np.ndarray:
    """
    Visualize predictions on an image
    
    Args:
        image_path (str): Path to the image or URL
        predictions (List[Dict]): List of predictions
        save_path (str, optional): Path to save visualized results
        
    Returns:
        np.ndarray: Image with visualized predictions
    """
    # Load image
    if image_path.startswith(('http://', 'https://')):
        # Load image from URL
        import urllib.request
        with urllib.request.urlopen(image_path) as resp:
            image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    else:
        # Load local image
        image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    # Colors for different classes (HSV to get bright colors)
    class_colors = {}
    
    # Print predictions
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        # Handle prediction format
        class_name = pred.get("class_name", "Unknown")
        confidence = pred.get("confidence", 0)
        box = pred.get("box", [])
        
        print(f"  {i+1}. {class_name} (confidence: {confidence:.2f})")
        print(f"     Box coordinates: {box}")
    
    # Visualize predictions
    for i, pred in enumerate(predictions):
        # Get prediction data
        class_name = pred.get("class_name", "Unknown")
        confidence = pred.get("confidence", 0)
        box = pred.get("box", [])
        
        if len(box) != 4:
            continue
        
        # Get box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get or generate color for class
        if class_name not in class_colors:
            # Generate color for class in HSV and convert to BGR
            hue = (hash(class_name) % 256) / 255.0
            saturation = 0.7
            value = 0.9
            
            # Convert HSV to BGR
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            class_colors[class_name] = bgr
        
        color = class_colors[class_name]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label with class name and confidence
        label = f"{class_name} {confidence:.2f}"
        
        # Get text size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background for text
        cv2.rectangle(
            image, 
            (x1, y1 - label_height - baseline - 5), 
            (x1 + label_width, y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Save image if path is provided
    if save_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, image)
        print(f"\nVisualization saved to: {save_path}")
    
    return image

def test_sagemaker_endpoint(endpoint_name: str, image_path: str, save_path: str = None, region: str = None) -> bool:
    """
    Test SageMaker endpoint with an image using AWS authentication
    
    Args:
        endpoint_name (str): SageMaker endpoint name or full URL
        image_path (str): Path to the image or URL
        save_path (str, optional): Path to save visualized results
        region (str, optional): AWS region
        
    Returns:
        bool: True if test was successful
    """
    try:
        # Extract endpoint name if full URL is provided
        if endpoint_name.startswith('https://'):
            # Extract endpoint name from URL like:
            # https://runtime.sagemaker.us-east-2.amazonaws.com/endpoints/endpoint-name/invocations
            endpoint_parts = endpoint_name.split('/')
            for i, part in enumerate(endpoint_parts):
                if part == 'endpoints' and i+1 < len(endpoint_parts):
                    endpoint_name = endpoint_parts[i+1]
                    break
        
        # Get AWS region
        if not region:
            region = os.environ.get('AWS_REGION')
            if not region:
                # Extract region from endpoint URL if possible
                if endpoint_name.startswith('https://') and 'sagemaker.' in endpoint_name:
                    parts = endpoint_name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'sagemaker' and i+1 < len(parts):
                            region = parts[i+1]
                            break
                
                if not region:
                    print("Error: AWS region not provided and not found in environment")
                    print("Please specify the region with --region or set AWS_REGION environment variable")
                    return False
        
        # Load image
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            import urllib.request
            with urllib.request.urlopen(image_path) as resp:
                image_data = resp.read()
        else:
            # Load local image
            with open(image_path, 'rb') as f:
                image_data = f.read()
        
        # Create SageMaker runtime client with proper authentication
        sm_runtime = boto3.client(
            'sagemaker-runtime',
            region_name=region,
            **get_aws_credentials()
        )
        
        # Send request to endpoint with proper AWS authentication
        print(f"Invoking SageMaker endpoint: {endpoint_name}")
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/octet-stream',
            Body=image_data
        )
        
        # Parse response
        response_body = response['Body'].read().decode('utf-8')
        response_data = json.loads(response_body)
        
        print(f"Raw response: {response_data}")  # Debug output
        
        # Extract predictions from response
        if isinstance(response_data, dict) and 'predictions' in response_data:
            predictions = response_data['predictions']
            print(f"Received {len(predictions)} predictions")
        else:
            print(f"Warning: Unexpected response format: {type(response_data)}")
            predictions = []
        
        # Visualize predictions
        image = visualize_predictions(image_path, predictions, save_path)
        if image is None:
            return False
        
        return True
        
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_http_endpoint(url: str, image_path: str, save_path: str = None) -> bool:
    """
    Test HTTP endpoint with an image (without AWS authentication)
    
    Args:
        url (str): Endpoint URL
        image_path (str): Path to the image or URL
        save_path (str, optional): Path to save visualized results
        
    Returns:
        bool: True if test was successful
    """
    try:
        # Load image
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            import urllib.request
            with urllib.request.urlopen(image_path) as resp:
                image_data = resp.read()
        else:
            # Load local image
            with open(image_path, 'rb') as f:
                image_data = f.read()
        
        # Send request to endpoint
        print(f"Sending request to endpoint: {url}")
        response = requests.post(
            url,
            data=image_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        # Check response
        if response.status_code != 200:
            print(f"Error: Endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Parse predictions
        response_data = response.json()
        print(f"Raw response: {response_data}")  # Debug output
        
        # Extract predictions from response
        if isinstance(response_data, list):
            # Direct list of predictions
            predictions = response_data
            print(f"Received {len(predictions)} predictions")
        elif isinstance(response_data, dict) and 'predictions' in response_data:
            # Dictionary with predictions key
            predictions = response_data['predictions']
            print(f"Received {len(predictions)} predictions")
        else:
            print(f"Warning: Unexpected response format: {type(response_data)}")
            predictions = []
        
        # Visualize predictions
        image = visualize_predictions(image_path, predictions, save_path)
        if image is None:
            return False
        
        return True
        
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SageMaker endpoint with an image")
    parser.add_argument("--url", help="Endpoint URL or SageMaker endpoint name")
    parser.add_argument("--endpoint", help="SageMaker endpoint name (alternative to --url)")
    parser.add_argument("--image", required=True, help="Path to the image or URL")
    parser.add_argument("--save", default="output/result.jpg", help="Path to save visualized results")
    parser.add_argument("--region", help="AWS region for SageMaker endpoint")
    parser.add_argument("--no-auth", action="store_true", help="Use direct HTTP without AWS authentication")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.url and not args.endpoint:
        print("Error: Either --url or --endpoint must be provided")
        return False
    
    endpoint_name = args.endpoint or args.url
    
    # Determine if this is a local/HTTP endpoint or SageMaker endpoint
    is_local_endpoint = (
        args.no_auth or 
        (args.url and (
            args.url.startswith('http://localhost') or 
            args.url.startswith('http://127.0.0.1') or
            args.url.startswith('http://0.0.0.0') or
            (args.url.startswith('http://') and not 'sagemaker' in args.url)
        ))
    )
    
    # Test endpoint
    if is_local_endpoint:
        if not args.url:
            print("Error: --url is required for local endpoints")
            return False
        print(f"Detected local endpoint, using HTTP requests")
        success = test_http_endpoint(args.url, args.image, args.save)
    else:
        print(f"Detected SageMaker endpoint, using AWS SDK")
        success = test_sagemaker_endpoint(endpoint_name, args.image, args.save, args.region)
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
