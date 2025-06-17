#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import boto3
import datetime
from typing import Dict, Any
from dotenv import load_dotenv
import time
import traceback

# Import utility modules
from utils import (
    load_config, 
    get_aws_credentials, 
    get_aws_region, 
    get_sagemaker_role_arn, 
    get_s3_bucket,
    download_model_from_wandb,
    upload_model_to_s3,
    run_command,
    run_command_realtime,
    prepare_model_for_sagemaker
)

# Load environment variables from .env file
load_dotenv()

def check_cloudwatch_logs_permissions(role_arn: str, region: str) -> bool:
    """
    Check if the SageMaker role has permissions to write to CloudWatch Logs
    
    Args:
        role_arn (str): SageMaker role ARN
        region (str): AWS region
        
    Returns:
        bool: True if permissions are sufficient, False otherwise
    """
    try:
        print(f"Checking CloudWatch Logs permissions for role: {role_arn}")
        
        # Extract role name from ARN
        role_name = role_arn.split('/')[-1]
        
        # Create IAM client
        iam_client = boto3.client('iam', region_name=region, **get_aws_credentials())
        
        # Check attached policies
        attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
        
        # Look for CloudWatch Logs policies in attached policies
        has_cloudwatch_policy = False
        for policy in attached_policies.get('AttachedPolicies', []):
            policy_arn = policy.get('PolicyArn', '')
            policy_name = policy.get('PolicyName', '')
            
            if ('CloudWatchLogs' in policy_name or 
                'SageMakerFullAccess' in policy_name or 
                'AdministratorAccess' in policy_name):
                has_cloudwatch_policy = True
                print(f"Found policy with possible CloudWatch Logs permissions: {policy_name}")
        
        # Check inline policies
        inline_policies = iam_client.list_role_policies(RoleName=role_name)
        
        for policy_name in inline_policies.get('PolicyNames', []):
            policy_document = iam_client.get_role_policy(
                RoleName=role_name,
                PolicyName=policy_name
            )
            
            # Check if policy has CloudWatch Logs permissions
            policy_json = policy_document.get('PolicyDocument', {})
            statements = policy_json.get('Statement', [])
            
            for statement in statements:
                effect = statement.get('Effect', '')
                actions = statement.get('Action', [])
                
                if isinstance(actions, str):
                    actions = [actions]
                
                if effect == 'Allow' and any('logs:' in action for action in actions):
                    has_cloudwatch_policy = True
                    print(f"Found inline policy with CloudWatch Logs permissions: {policy_name}")
        
        if has_cloudwatch_policy:
            print("✅ Role has policies that may allow CloudWatch Logs access")
            return True
        else:
            print("⚠️ Role may not have sufficient permissions for CloudWatch Logs")
            print("To add CloudWatch Logs permissions, run:")
            print(f"aws iam attach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess")
            return False
            
    except Exception as e:
        print(f"Error checking CloudWatch Logs permissions: {str(e)}")
        return False

def deploy_to_serverless_sagemaker(
    model_path: str,
    model_name: str,
    endpoint_name: str,
    role_arn: str,
    region: str,
    s3_bucket: str,
    instance_type: str = "ml.t2.medium",
    instance_count: int = 1,
    custom_container: bool = False,
    container_image: str = None,
    environment_vars: Dict[str, str] = None
) -> str:
    """
    Deploy a model to SageMaker serverless endpoint
    
    Args:
        model_path (str): Path to the model file
        model_name (str): Name of the model
        endpoint_name (str): Name of the endpoint
        role_arn (str): SageMaker role ARN
        region (str): AWS region
        s3_bucket (str): S3 bucket for model artifacts
        instance_type (str): Instance type for the endpoint
        instance_count (int): Number of instances
        custom_container (bool): Whether to use a custom container
        container_image (str): URI of the custom container image
        environment_vars (Dict[str, str]): Environment variables for the endpoint
        
    Returns:
        str: Endpoint name if successful, None otherwise
    """
    try:
        print(f"\nDeploying model to SageMaker serverless endpoint: {endpoint_name}")
        
        # Check CloudWatch Logs permissions
        if not check_cloudwatch_logs_permissions(role_arn, region):
            print("Warning: Role may not have sufficient permissions for CloudWatch Logs")
            print("This may affect logging and monitoring capabilities")
        
        # Create SageMaker client
        sagemaker_client = boto3.client('sagemaker', region_name=region, **get_aws_credentials())
        
        # Upload model to S3 if not already there
        if not model_path.startswith('s3://'):
            print(f"Uploading model to S3: {model_path}")
            model_path = upload_model_to_s3(model_path, s3_bucket, model_name, region)
        
        # Create model
        print(f"Creating model: {model_name}")
        model_arn = f"arn:aws:sagemaker:{region}:{role_arn.split(':')[4]}:model/{model_name}"
        
        if custom_container and container_image:
            # Use custom container
            print(f"Using custom container: {container_image}")
            model_response = sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': container_image,
                    'ModelDataUrl': model_path,
                    'Environment': environment_vars or {}
                },
                ExecutionRoleArn=role_arn
            )
        else:
            # Use default container
            print("Using default container")
            model_response = sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': f"246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
                    'ModelDataUrl': model_path,
                    'Environment': environment_vars or {}
                },
                ExecutionRoleArn=role_arn
            )
        
        # Create endpoint configuration
        print(f"Creating endpoint configuration: {endpoint_name}-config")
        endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=f"{endpoint_name}-config",
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count
                }
            ]
        )
        
        # Create endpoint
        print(f"Creating endpoint: {endpoint_name}")
        endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=f"{endpoint_name}-config"
        )
        
        # Don't wait for endpoint to be in service
        print(f"Endpoint creation initiated: {endpoint_response['EndpointArn']}")
        print("Check the AWS console or use AWS CLI to monitor the endpoint status.")
        return endpoint_response["EndpointArn"]
        
    except Exception as e:
        print(f"Error deploying model to SageMaker: {str(e)}")
        return None

def create_sagemaker_model(
    model_name: str,
    model_path: str,
    role_arn: str,
    custom_container: str = None
) -> bool:
    """
    Create a SageMaker model
    
    Args:
        model_name (str): Name of the model
        model_path (str): Path to the model artifact in W&B
        role_arn (str): SageMaker role ARN
        custom_container (str): URI of the custom container image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load configuration
        config = load_config()
        
        # Get AWS region
        region = config["aws"]["region"]
        
        # Get S3 bucket from config
        s3_bucket = config["aws"]["s3_bucket"]
        
        # Download model from W&B
        print("\nDownloading model from W&B...")
        local_model_path = download_model_from_wandb(
            artifact_path=model_path,
            version="latest"
        )
        
        if not local_model_path:
            print("Error: Failed to download model from W&B")
            return False
        
        # Upload model to S3
        print(f"Uploading model to S3...")
        model_s3_url = upload_model_to_s3(local_model_path, s3_bucket, model_name, region)
        
        if not model_s3_url:
            print("Error: Failed to upload model to S3")
            return False
        
        # Create SageMaker client
        sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Create model
        if custom_container:
            # Use custom container
            # Extract repository and tag from custom_container
            repository, tag = custom_container.split(':')
            model_response = sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'ContainerHostname': model_name,
                    'Image': f"{repository}:{tag}",
                    'ModelDataUrl': model_s3_url,
                    'Mode': 'SingleModel',
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'serve',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': region,
                        'MODEL_SERVER_TIMEOUT': '60',
                        'MODEL_SERVER_WORKERS': '1',
                        'MODEL_MAX_REQUEST_SIZE': '6291456'  # 6MB in bytes
                    }
                },
                ExecutionRoleArn=role_arn
            )
        else:
            # Use default container
            model_response = sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'ContainerHostname': model_name,
                    'Image': f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.0-gpu-py310",
                    'ModelDataUrl': model_s3_url,
                    'Mode': 'SingleModel',
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'serve',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': region,
                        'MODEL_SERVER_TIMEOUT': '60',
                        'MODEL_SERVER_WORKERS': '1',
                        'MODEL_MAX_REQUEST_SIZE': '6291456'  # 6MB in bytes
                    }
                },
                ExecutionRoleArn=role_arn
            )
        
        print(f"Model {model_name} created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return False

def deploy_serverless_endpoint(
    model_name: str,
    endpoint_name: str,
    sagemaker_client,
    serverless_config: dict
) -> str:
    """
    Deploy a serverless endpoint using the specified model
    
    Args:
        model_name (str): Name of the SageMaker model to deploy
        endpoint_name (str): Name for the endpoint
        sagemaker_client: Boto3 SageMaker client
        serverless_config (dict): Serverless configuration parameters
        
    Returns:
        str: Endpoint name if successful
    """
    try:
        print(f"\nDeploying serverless endpoint {endpoint_name}...")
        
        # Create endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config"
        
        # Get serverless configuration parameters from config
        max_concurrency = serverless_config["max_concurrency"]
        memory_size = serverless_config["memory_size"]
        
        print(f"Configuring serverless endpoint with:")
        print(f"- Max concurrency: {max_concurrency}")
        print(f"- Memory size: {memory_size}MB")
        
        # Create endpoint configuration with serverless settings
        endpoint_config = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size,
                        "MaxConcurrency": max_concurrency
                    }
                }
            ]
        }
        
        print("\nCreating endpoint configuration...")
        sagemaker_client.create_endpoint_config(**endpoint_config)
        
        # Create endpoint
        print("\nCreating endpoint...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        print(f"\nServerless endpoint {endpoint_name} deployment started!")
        print("Note: The endpoint will take some time to be ready for inference.")
        print("You can check the status using AWS Console or AWS CLI.")
        return endpoint_name
        
    except Exception as e:
        print(f"Error deploying serverless endpoint: {str(e)}")
        print(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker serverless endpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--model-path", type=str,
                        help="Path to the model file (local or W&B artifact)")
    parser.add_argument("--artifact-path", type=str,
                        help="Weights & Biases artifact path (entity/project/artifact_name)")
    parser.add_argument("--artifact-version", type=str, default="latest",
                        help="Weights & Biases artifact version")
    parser.add_argument("--model-name", type=str,
                        help="Name for the SageMaker model")
    parser.add_argument("--endpoint-name", type=str,
                        help="Name for the SageMaker endpoint")
    parser.add_argument("--memory-size", type=int,
                        help="Memory size for serverless endpoint in MB")
    parser.add_argument("--max-concurrency", type=int,
                        help="Maximum concurrency for serverless endpoint")
    parser.add_argument("--custom-container", action="store_true",
                        help="Use custom container for deployment")
    parser.add_argument("--container-image", type=str,
                        help="URI of the custom container image")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.debug:
        print("\nCommand line arguments:")
        print(f"config: {args.config}")
        print(f"model_path: {args.model_path}")
        print(f"artifact_path: {args.artifact_path}")
        print(f"artifact_version: {args.artifact_version}")
        print(f"model_name: {args.model_name}")
        print(f"endpoint_name: {args.endpoint_name}")
        print(f"memory_size: {args.memory_size}")
        print(f"max_concurrency: {args.max_concurrency}")
        print(f"custom_container: {args.custom_container}")
        print(f"container_image: {args.container_image}")
    
    # Load configuration
    config = load_config(args.config)
    
    if args.debug:
        print("\nConfiguration:")
        print(f"config: {config}")
    
    # Get AWS settings
    region = get_aws_region(config)
    s3_bucket = get_s3_bucket(config)
    role_arn = get_sagemaker_role_arn(config)
    
    if args.debug:
        print("\nAWS Settings:")
        print(f"region: {region}")
        print(f"s3_bucket: {s3_bucket}")
        print(f"role_arn: {role_arn}")
    
    # Get model settings
    model_name = args.model_name or config.get("model", {}).get("name", "yolov8-model")
    
    # Get endpoint name from config or generate one
    endpoint_name = args.endpoint_name or config.get("serverless", {}).get("name")
    if not endpoint_name:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"{model_name}-serverless-endpoint"
        if args.debug:
            print(f"\nGenerated endpoint name: {endpoint_name}")
    
    # Get serverless configuration
    serverless_config = config.get("serverless", {})
    memory_size = args.memory_size or serverless_config.get("memory_size", 2048)
    max_concurrency = args.max_concurrency or serverless_config.get("max_concurrency", 10)
    enable_provisioned_concurrency = serverless_config.get("enable_provisioned_concurrency", False)
    provisioned_concurrency = serverless_config.get("provisioned_concurrency", 1)
    
    # Get container settings
    container_config = config.get("container", {})
    use_custom_container = args.custom_container or container_config.get("use_custom", False)
    container_image = args.container_image or container_config.get("image") or container_config.get("custom_image")
    
    if use_custom_container and not container_image:
        print("Error: Custom container is enabled but no container image is specified")
        print("Please provide a container image URI using --container-image or in config.yaml")
        sys.exit(1)
    
    # Get environment variables
    environment_vars = config.get("environment", {})
    environment_vars["MODEL_PATH"] = "/opt/ml/model/model.pt"
    
    if args.debug:
        print("\nDeployment Settings:")
        print(f"model_name: {model_name}")
        print(f"endpoint_name: {endpoint_name}")
        print(f"memory_size: {memory_size}")
        print(f"max_concurrency: {max_concurrency}")
        print(f"enable_provisioned_concurrency: {enable_provisioned_concurrency}")
        print(f"provisioned_concurrency: {provisioned_concurrency}")
        print(f"use_custom_container: {use_custom_container}")
        print(f"container_image: {container_image}")
        print(f"environment_vars: {environment_vars}")
    
    # Check if model S3 URI is provided
    model_s3_uri = config.get("model", {}).get("s3_uri")
    if model_s3_uri:
        print(f"Using provided model S3 URI: {model_s3_uri}")
    else:
        # Download model from Weights & Biases if artifact path is specified
        artifact_path = args.artifact_path or config.get("model", {}).get("artifact_path")
        if artifact_path:
            version = args.artifact_version or config.get("model", {}).get("version", "latest")
            print(f"Downloading model from Weights & Biases: {artifact_path}:{version}")
            model_path = download_model_from_wandb(artifact_path, version)
            print(f"Model downloaded to: {model_path}")
            
            # Prepare model for SageMaker
            print(f"Preparing model for SageMaker: {model_path}")
            prepared_model_path = prepare_model_for_sagemaker(model_path, model_name)
            print(f"Model prepared: {prepared_model_path}")
            
            # Upload model to S3
            print(f"Uploading model to S3")
            model_s3_uri = upload_model_to_s3(prepared_model_path, s3_bucket, model_name, region)
            print(f"Model uploaded to: {model_s3_uri}")
        else:
            print("Error: No model artifact path specified")
            sys.exit(1)
    
    if not model_s3_uri:
        print("Error: Failed to get model S3 URI")
        sys.exit(1)
    
    # Deploy to serverless endpoint
    endpoint_arn = deploy_to_serverless_endpoint(
        model_name=model_name,
        endpoint_name=endpoint_name,
        role_arn=role_arn,
        region=region,
        s3_bucket=s3_bucket,
        model_s3_uri=model_s3_uri,
        memory_size=memory_size,
        max_concurrency=max_concurrency,
        enable_provisioned_concurrency=enable_provisioned_concurrency,
        provisioned_concurrency=provisioned_concurrency,
        custom_container=use_custom_container,
        container_image=container_image,
        environment_vars=environment_vars
    )
    
    if endpoint_arn:
        print("\nDeployment successful!")
        print(f"Endpoint ARN: {endpoint_arn}")
        print(f"Endpoint Name: {endpoint_name}")
        print("\nTo test the endpoint, run:")
        print(f"aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} --content-type application/json --body '{{\"data\": \"test\"}}' output.json")
    else:
        print("\nDeployment failed!")
        sys.exit(1)

def deploy_to_serverless_endpoint(
    model_name: str,
    endpoint_name: str,
    role_arn: str,
    region: str,
    s3_bucket: str,
    model_s3_uri: str,
    memory_size: int = 2048,
    max_concurrency: int = 10,
    enable_provisioned_concurrency: bool = False,
    provisioned_concurrency: int = 1,
    custom_container: bool = False,
    container_image: str = None,
    environment_vars: Dict[str, str] = None
) -> str:
    """
    Deploy model to SageMaker serverless endpoint
    """
    try:
        print(f"\nDeploying model to SageMaker serverless endpoint: {endpoint_name}")
        
        # Create SageMaker client
        sagemaker_client = boto3.client("sagemaker", region_name=region, **get_aws_credentials())
        
        # Create model
        print(f"Creating model: {model_name}")
        
        # Define container configuration
        container_config = {
            "ModelDataUrl": model_s3_uri,
            "Environment": environment_vars or {}
        }
        
        if custom_container and container_image:
            print(f"Using custom container: {container_image}")
            container_config["Image"] = container_image
        else:
            print("Using default container")
            container_config["Image"] = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.0-gpu-py310"
        
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer=container_config,
            ExecutionRoleArn=role_arn
        )
        
        # Create serverless endpoint configuration
        endpoint_config_name = f"{endpoint_name}-config"
        print(f"Creating serverless endpoint configuration: {endpoint_config_name}")
        endpoint_config = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size,
                        "MaxConcurrency": max_concurrency
                    }
                }
            ]
        }
        
        # Add provisioned concurrency if enabled
        if enable_provisioned_concurrency:
            endpoint_config["ProductionVariants"][0]["ServerlessConfig"]["ProvisionedConcurrency"] = provisioned_concurrency
        
        endpoint_config_response = sagemaker_client.create_endpoint_config(**endpoint_config)
        
        # Create endpoint
        print(f"Creating endpoint: {endpoint_name}")
        endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        # Don't wait for endpoint to be in service
        print(f"Endpoint creation initiated: {endpoint_response['EndpointArn']}")
        print("Check the AWS console or use AWS CLI to monitor the endpoint status.")
        return endpoint_response["EndpointArn"]
        
    except Exception as e:
        print(f"Error deploying to serverless endpoint: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 