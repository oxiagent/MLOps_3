#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime

# Import utility modules
from utils import (
    download_model_from_wandb,
    load_config,
    run_command,
    run_command_realtime,
    prepare_model_for_sagemaker
)

# Define base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOCKERFILE_DIR = os.path.join(SCRIPT_DIR, "inference_container")

def build_docker_image(
    dockerfile_dir: str,
    image_name: str,
    image_tag: str,
    build_args: dict = None
) -> bool:
    """
    Build Docker image
    
    Args:
        dockerfile_dir (str): Directory containing Dockerfile
        image_name (str): Name of the image
        image_tag (str): Tag for the image
        build_args (dict): Additional build arguments
        
    Returns:
        bool: True if build was successful
    """
    # If image_name already contains a tag, use it as is
    if ":" in image_name:
        full_image_name = image_name
    else:
        full_image_name = f"{image_name}:{image_tag}"
    
    print(f"\nBuilding Docker image: {full_image_name}")
    
    # Authenticate to AWS ECR for deep learning containers
    try:
        print("Authenticating to AWS Deep Learning Containers...")
        ecr_repo = "763104351884.dkr.ecr.us-east-1.amazonaws.com"
        aws_cmd = f"aws ecr get-login-password --region us-east-1"
        
        aws_result = run_command(aws_cmd, capture_output=True)
        if aws_result.returncode != 0:
            print(f"Error getting ECR login password: {aws_result.stderr}")
        else:
            password = aws_result.stdout.strip()
            docker_cmd = f'echo "{password}" | docker login --username AWS --password-stdin {ecr_repo}'
            docker_result = run_command(docker_cmd, capture_output=True)
            
            if docker_result.returncode != 0:
                print(f"Error logging in to ECR: {docker_result.stderr}")
            else:
                print("Successfully authenticated to AWS Deep Learning Containers")
    except Exception as e:
        print(f"Warning: Could not authenticate to AWS ECR: {e}")
        print("Proceeding with build anyway, in case the image is cached locally...")
    
    # Construct build command
    build_cmd = f"docker build -t {full_image_name} {dockerfile_dir}"
    
    # Add build arguments if provided
    if build_args:
        for key, value in build_args.items():
            build_cmd += f" --build-arg {key}={value}"
    
    # Run build command
    try:
        run_command_realtime(build_cmd)
        return True
    except Exception as e:
        print(f"Error building Docker image: {e}")
        return False

def run_container(
    image_name: str,
    image_tag: str,
    model_dir: str,
    model_filename: str
) -> bool:
    """
    Run Docker container locally
    
    Args:
        image_name (str): Name of the image
        image_tag (str): Tag of the image
        model_dir (str): Path to model directory
        model_filename (str): Name of the model file
        
    Returns:
        bool: True if container started successfully
    """
    if ":" in image_name:
        full_image_name = image_name
    else:
        full_image_name = f"{image_name}:{image_tag}"
    
    print(f"\nRunning container: {full_image_name}")
    
    try:
        container_name = f"{image_name.replace('/', '_').replace(':', '_')}_local"
        
        # Ensure model_dir is a directory, not a file
        if os.path.isfile(model_dir):
            # If model_dir is actually a file path, get its directory and filename
            model_path = model_dir
            model_dir = os.path.dirname(model_path)
            model_target = "/opt/ml/model"
        else:
            # If model_dir is a directory, mount the entire directory
            model_target = "/opt/ml/model"
        
        run_cmd = (
            f"docker run --rm --name {container_name} "
            f"-p 8080:8080 "
            f"-v {model_dir}:{model_target} "
            f"-e MODEL_NAME={model_filename} "
            f"-e SAGEMAKER_PROGRAM=app.py "
            f"{full_image_name}"
        )
        
        print(f"Running command: {run_cmd}")
        run_command_realtime(run_cmd)
        return True
    except Exception as e:
        print(f"Error running container: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build and run Docker container locally")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--dockerfile-dir", default=DEFAULT_DOCKERFILE_DIR, help="Path to Dockerfile directory")
    parser.add_argument("--image-name", help="Name for the Docker image")
    parser.add_argument("--image-tag", default="latest", help="Tag for the Docker image")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = download_model_from_wandb(
            config["model"]["artifact_path"],
            config["model"].get("version", "v2")
        )
    
    # Prepare model for SageMaker
    model_dir = prepare_model_for_sagemaker(model_path)
    model_filename = config["model"]["local_file"]
    
    # Ensure model_dir is a directory
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)
    
    # Get image name
    if args.image_name:
        image_name = args.image_name
    else:
        image_name = config["container"].get("custom_image", config["container"]["default_image_name"])
    
    # Build Docker image
    if not args.skip_build:
        build_args = {
            "MODEL_DIR": os.path.dirname(model_dir),
            "MODEL_NAME": model_filename
        }
        
        if not build_docker_image(
            args.dockerfile_dir,
            image_name,
            args.image_tag,
            build_args
        ):
            sys.exit(1)
    
    # Run container
    if not run_container(image_name, args.image_tag, model_dir, model_filename):
        sys.exit(1)

if __name__ == "__main__":
    main() 