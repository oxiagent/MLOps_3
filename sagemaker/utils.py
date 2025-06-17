#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import boto3
import wandb
import tarfile
import shutil
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# Define __all__ to explicitly specify which functions should be imported with from utils import *
__all__ = [
    'load_config', 
    'get_aws_credentials', 
    'get_aws_region', 
    'get_sagemaker_role_arn', 
    'get_s3_bucket',
    'download_model_from_wandb',
    'upload_model_to_s3',
    'find_model_file',
    'run_command',
    'run_command_realtime',
    'prepare_model_for_sagemaker'
]

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def get_aws_credentials() -> Dict[str, str]:
    """
    Get AWS credentials from environment variables or AWS CLI config
    
    Returns:
        Dict[str, str]: AWS credentials or empty dict if using CLI profile
    """
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # Use environment variables if they are defined and not empty
    if aws_access_key and aws_secret_key and aws_access_key != "" and aws_secret_key != "":
        return {
            "aws_access_key_id": aws_access_key,
            "aws_secret_access_key": aws_secret_key
        }
    
    # Otherwise use AWS CLI configuration
    print("Using AWS CLI credentials from your profile")
    return {}  # Empty dict means boto3 will use settings from ~/.aws/credentials

def get_aws_region(config: Dict[str, Any]) -> str:
    """
    Get AWS region from config or environment variable
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: AWS region
    """
    region = config.get("aws", {}).get("region")
    if not region:
        region = os.getenv("AWS_REGION", "us-east-1")
    return region

def get_sagemaker_role_arn(config: Dict[str, Any]) -> str:
    """
    Get SageMaker role ARN from config or environment variable
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: SageMaker role ARN
    """
    role_arn = config.get("aws", {}).get("sagemaker_role_arn")
    if not role_arn:
        role_arn = os.getenv("AWS_SAGEMAKER_ROLE_ARN")
    
    if not role_arn:
        print("Error: SageMaker role ARN not found")
        print("Please set AWS_SAGEMAKER_ROLE_ARN environment variable or specify it in config.yaml")
        sys.exit(1)
    
    return role_arn

def get_s3_bucket(config: Dict[str, Any]) -> str:
    """
    Get S3 bucket name from config or environment variable
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: S3 bucket name
    """
    bucket = config.get("aws", {}).get("s3_bucket")
    if not bucket:
        bucket = os.getenv("AWS_S3_BUCKET")
    
    if not bucket:
        print("Error: S3 bucket name not found")
        print("Please set AWS_S3_BUCKET environment variable or specify it in config.yaml")
        sys.exit(1)
    
    return bucket

def download_model_from_wandb(artifact_path: str, version: str = "v2") -> str:
    """
    Download model from Weights & Biases
    
    Args:
        artifact_path (str): Path to the model artifact
        version (str): Version of the model to download
        
    Returns:
        str: Path to the downloaded model
    """
    try:
        # Load config
        config = load_config()
        
        # Initialize wandb with config values
        wandb.init(
            project=config["wandb"]["project"],
            job_type=config["wandb"]["job_type"]
        )
        
        # Download the artifact
        artifact = wandb.use_artifact(f"{artifact_path}:{version}")
        artifact_dir = artifact.download()
        
        # Find the model file
        model_path = find_model_file(artifact_dir)
        if not model_path:
            raise ValueError(f"No model file found in {artifact_dir}")
        
        return model_path
    except Exception as e:
        print(f"Error downloading model from wandb: {e}")
        sys.exit(1)

def upload_model_to_s3(model_path: str, bucket_name: str, model_name: str, region: str) -> str:
    """
    Upload model to S3 with proper SageMaker directory structure
    
    Args:
        model_path (str): Path to the model file
        bucket_name (str): S3 bucket name
        model_name (str): Name of the model
        region (str): AWS region
        
    Returns:
        str: S3 path to the uploaded model
    """
    try:
        s3_client = boto3.client("s3", region_name=region, **get_aws_credentials())
        
        # Create a temporary directory for proper SageMaker model structure
        temp_dir = os.path.join(os.path.dirname(model_path), f"temp_sagemaker_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy model file to temp directory with .pt extension
        model_file = os.path.join(temp_dir, "model.pt")
        print(f"Creating SageMaker model structure: {model_path} -> {model_file}")
        shutil.copy(model_path, model_file)
        
        # Add metadata file (optional)
        with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
            f.write('{"model_type": "yolov8", "source": "' + os.path.basename(model_path) + '"}')
        
        # Create tar.gz archive with the model at the root level
        tar_path = os.path.join(os.path.dirname(temp_dir), f"{model_name}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add model file with explicit arcname at the root level
            tar.add(os.path.join(temp_dir, "model.pt"), arcname="model.pt")
            # Add metadata file at the root level
            tar.add(os.path.join(temp_dir, "metadata.json"), arcname="metadata.json")
        
        # Upload to S3
        s3_key = f"models/{model_name}.tar.gz"
        print(f"Uploading model to S3: {tar_path} -> s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(tar_path, bucket_name, s3_key)
        
        # Clean up
        shutil.rmtree(temp_dir)
        os.remove(tar_path)
        
        print(f"Model successfully uploaded to: s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        print(f"Error uploading model to S3: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def find_model_file(model_dir: str, supported_extensions: List[str] = None) -> str:
    """
    Find model file in directory
    
    Args:
        model_dir (str): Directory to search in
        supported_extensions (List[str]): List of supported file extensions
        
    Returns:
        str: Path to the model file
    """
    if supported_extensions is None:
        supported_extensions = [".pt", ".pth", ".onnx", ".pb"]
    
    for root, _, files in os.walk(model_dir):
        for file in files:
            if any(file.endswith(ext) for ext in supported_extensions):
                return os.path.join(root, file)
    
    return ""

def run_command(
    command: Union[str, List[str]],
    check: bool = True,
    shell: bool = True,
    capture_output: bool = True,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """
    Run a command and return the result
    
    Args:
        command: Command to run (string or list of strings)
        check: Whether to raise an exception on non-zero exit code
        shell: Whether to run the command through the shell
        capture_output: Whether to capture stdout and stderr
        timeout: Optional timeout in seconds
        
    Returns:
        subprocess.CompletedProcess object
    """
    print(f"Running command: {command}")
    
    try:
        if capture_output:
            result = subprocess.run(
                command,
                shell=shell,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0 and check:
                print(f"Command failed with code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stdout, result.stderr
                )
        else:
            result = subprocess.run(
                command,
                shell=shell,
                check=check,
                text=True,
                timeout=timeout
            )
        
        return result
    except Exception as e:
        print(f"Error executing command: {e}")
        raise

def run_command_realtime(
    command: Union[str, List[str]],
    shell: bool = True,
    timeout: Optional[int] = None
) -> int:
    """
    Run a command with real-time output display
    
    Args:
        command: Command to run (string or list of strings)
        shell: Whether to run the command through the shell
        timeout: Optional timeout in seconds
        
    Returns:
        int: Return code of the command
    """
    print(f"Running command with real-time output: {command}")
    print("-" * 80)
    
    try:
        if shell:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
        else:
            process = subprocess.Popen(
                command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
        
        def read_output(pipe, prefix=""):
            for line in iter(pipe.readline, ''):
                print(f"{prefix}{line.rstrip()}", flush=True)
        
        # Read output in a separate thread
        output_thread = threading.Thread(
            target=read_output,
            args=(process.stdout,)
        )
        output_thread.daemon = True
        output_thread.start()
        
        # Wait for process to complete
        return_code = process.wait(timeout=timeout)
        print("-" * 80)
        print(f"Command completed with return code: {return_code}")
        
        return return_code
    except Exception as e:
        print(f"Error executing command: {e}")
        raise

def prepare_model_for_sagemaker(model_path: str, model_name: str = None) -> str:
    """
    Prepare model for SageMaker deployment with proper SageMaker directory structure
    
    Args:
        model_path (str): Path to the model file
        model_name (str): Name of the model (optional)
        
    Returns:
        str: Path to the prepared model file (not directory)
    """
    try:
        # Load config
        config = load_config()
        
        if model_name is None:
            model_name = config["model"].get("name", "model")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config["model"]["local_dir"])
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy model file directly to models directory with proper name
        target_path = os.path.join(models_dir, f"{model_name}.pt")
        print(f"Preparing model: {model_path} -> {target_path}")
        shutil.copy(model_path, target_path)
        
        print(f"Model prepared for SageMaker: {target_path}")
        return target_path
    except Exception as e:
        print(f"Error preparing model for SageMaker: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 