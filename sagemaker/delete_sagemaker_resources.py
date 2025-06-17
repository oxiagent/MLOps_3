#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import boto3
import time
from dotenv import load_dotenv

# Import utility modules
from utils import (
    load_config, 
    get_aws_credentials, 
    get_aws_region
)

# Load environment variables from .env file
load_dotenv()

def delete_sagemaker_resources(
    model_name: str, 
    region: str, 
    config: dict,
    wait_for_deletion: bool = False,
    delete_model: bool = True,
    delete_endpoint_config: bool = True,
    delete_endpoint: bool = True,
    endpoint_name: str = None,
    endpoint_config_name: str = None
):
    """
    Delete SageMaker resources (model, endpoint config, endpoint) if they exist
    
    Args:
        model_name (str): Name of the SageMaker model
        region (str): AWS region
        config (dict): Configuration dictionary
        wait_for_deletion (bool): Whether to wait for endpoint deletion to complete
        delete_model (bool): Whether to delete the model
        delete_endpoint_config (bool): Whether to delete the endpoint config
        delete_endpoint (bool): Whether to delete the endpoint
        endpoint_name (str): Specific endpoint name to delete (overrides config)
        endpoint_config_name (str): Specific endpoint config name to delete (overrides config)
    """
    print(f"\n=== Deleting SageMaker resources for model: {model_name} ===")
    
    # Create SageMaker client
    sagemaker_client = boto3.client("sagemaker", region_name=region, **get_aws_credentials())
    
    # Get endpoint and config names from args or config
    if endpoint_name is None:
        endpoint_name = config.get("serverless", {}).get("name", f"{model_name}-serverless-endpoint")
    
    if endpoint_config_name is None:
        endpoint_config_name = config.get("serverless", {}).get("config_name", f"{endpoint_name}-config")
    
    print(f"Using endpoint name: {endpoint_name}")
    print(f"Using endpoint config name: {endpoint_config_name}")
    
    # Check for endpoint and delete if exists
    if delete_endpoint:
        try:
            # Check if endpoint exists
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            print(f"Found existing endpoint: {endpoint_name}")
            print(f"Status: {endpoint_info.get('EndpointStatus', 'Unknown')}")
            print(f"Created: {endpoint_info.get('CreationTime', 'Unknown')}")
            print(f"Deleting endpoint...")
            
            # Delete endpoint
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Endpoint deletion initiated: {endpoint_name}")
            
            if wait_for_deletion:
                print("Waiting for endpoint deletion to complete...")
                waiter = sagemaker_client.get_waiter('endpoint_deleted')
                waiter.wait(EndpointName=endpoint_name)
                print("Endpoint deletion completed")
            else:
                print("Endpoint deletion initiated in background")
                # Sleep a bit to let the deletion start
                time.sleep(5)
            
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                print(f"No existing endpoint found with name: {endpoint_name}")
            else:
                print(f"Error checking endpoint: {str(e)}")
    
    # Check for endpoint config and delete if exists
    if delete_endpoint_config:
        try:
            # Check if endpoint config exists
            config_info = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            print(f"Found existing endpoint configuration: {endpoint_config_name}")
            print(f"Created: {config_info.get('CreationTime', 'Unknown')}")
            print(f"Deleting endpoint configuration...")
            
            # Delete endpoint config
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            print(f"Endpoint configuration deleted: {endpoint_config_name}")
            
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint configuration" in str(e):
                print(f"No existing endpoint configuration found with name: {endpoint_config_name}")
            else:
                print(f"Error checking endpoint configuration: {str(e)}")
    
    # Check for model and delete if exists
    if delete_model:
        try:
            # Check if model exists
            model_info = sagemaker_client.describe_model(ModelName=model_name)
            print(f"Found existing model: {model_name}")
            print(f"Container: {model_info.get('PrimaryContainer', {}).get('Image', 'Unknown')}")
            print(f"Deleting model...")
            
            # Delete model
            sagemaker_client.delete_model(ModelName=model_name)
            print(f"Model deleted: {model_name}")
            
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find model" in str(e):
                print(f"No existing model found with name: {model_name}")
            else:
                print(f"Error checking model: {str(e)}")

def list_sagemaker_resources(region: str):
    """
    List all SageMaker resources in the specified region
    
    Args:
        region (str): AWS region
    """
    print(f"\n=== Listing SageMaker Resources in {region} ===")
    
    # Create SageMaker client
    sagemaker_client = boto3.client("sagemaker", region_name=region, **get_aws_credentials())
    
    # List models
    print("\nModels:")
    try:
        models = sagemaker_client.list_models(MaxResults=100)
        if not models.get('Models'):
            print("  No models found")
        else:
            for model in models.get('Models', []):
                print(f"  - {model.get('ModelName')}")
                print(f"    Created: {model.get('CreationTime')}")
    except Exception as e:
        print(f"  Error listing models: {str(e)}")
    
    # List endpoints
    print("\nEndpoints:")
    try:
        endpoints = sagemaker_client.list_endpoints(MaxResults=100)
        if not endpoints.get('Endpoints'):
            print("  No endpoints found")
        else:
            for endpoint in endpoints.get('Endpoints', []):
                print(f"  - {endpoint.get('EndpointName')}")
                print(f"    Status: {endpoint.get('EndpointStatus')}")
                print(f"    Created: {endpoint.get('CreationTime')}")
    except Exception as e:
        print(f"  Error listing endpoints: {str(e)}")
    
    # List endpoint configs
    print("\nEndpoint Configurations:")
    try:
        endpoint_configs = sagemaker_client.list_endpoint_configs(MaxResults=100)
        if not endpoint_configs.get('EndpointConfigs'):
            print("  No endpoint configurations found")
        else:
            for config in endpoint_configs.get('EndpointConfigs', []):
                print(f"  - {config.get('EndpointConfigName')}")
                print(f"    Created: {config.get('CreationTime')}")
    except Exception as e:
        print(f"  Error listing endpoint configurations: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Delete SageMaker resources (model, endpoint config, endpoint)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--model-name", type=str,
                        help="Name for the SageMaker model to delete")
    parser.add_argument("--endpoint-name", type=str,
                        help="Name for the SageMaker endpoint to delete (overrides config)")
    parser.add_argument("--endpoint-config-name", type=str,
                        help="Name for the SageMaker endpoint config to delete (overrides config)")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for deletion to complete")
    parser.add_argument("--list", action="store_true",
                        help="List all SageMaker resources")
    parser.add_argument("--no-model", action="store_true",
                        help="Do not delete the model")
    parser.add_argument("--no-config", action="store_true",
                        help="Do not delete the endpoint configuration")
    parser.add_argument("--no-endpoint", action="store_true",
                        help="Do not delete the endpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get AWS region
    region = get_aws_region(config)
    
    # List resources if requested
    if args.list:
        list_sagemaker_resources(region)
        return
    
    # Get model name
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = config.get("model", {}).get("name")
        if not model_name:
            print("Error: Model name not specified")
            print("Please specify it either in config.yaml or with --model-name argument")
            sys.exit(1)
    
    # Delete resources
    delete_sagemaker_resources(
        model_name=model_name,
        region=region,
        config=config,
        wait_for_deletion=args.wait,
        delete_model=not args.no_model,
        delete_endpoint_config=not args.no_config,
        delete_endpoint=not args.no_endpoint,
        endpoint_name=args.endpoint_name,
        endpoint_config_name=args.endpoint_config_name
    )
    
    print("\nDeletion completed!")
    print(f"Model name: {model_name}")
    if args.endpoint_name:
        print(f"Endpoint name: {args.endpoint_name}")
    if args.endpoint_config_name:
        print(f"Endpoint config name: {args.endpoint_config_name}")
    print(f"Region: {region}")

if __name__ == "__main__":
    main() 