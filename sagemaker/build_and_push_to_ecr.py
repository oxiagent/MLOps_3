#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import subprocess
import time
from utils import load_config, run_command, run_command_realtime, get_aws_region

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def login_to_ecr(region):
    """Login to ECR using the AWS CLI."""
    logger.info(f"Logging in to ECR in region {region}")
    
    # Get AWS account ID
    account_id_cmd = ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]
    try:
        account_id = subprocess.check_output(account_id_cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get AWS account ID: {e}")
        sys.exit(1)
    
    # Get ECR login password
    password_cmd = ["aws", "ecr", "get-login-password", "--region", region]
    try:
        password = subprocess.check_output(password_cmd, text=True).strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get ECR login password: {e}")
        sys.exit(1)
    
    # Construct the ECR registry URL and login
    registry_url = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    logger.info(f"Logging in to ECR registry: {registry_url}")
    
    login_cmd = ["docker", "login", "--username", "AWS", "--password-stdin", registry_url]
    try:
        process = subprocess.Popen(login_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=password)
        
        if process.returncode != 0:
            logger.error(f"Docker login failed: {stderr}")
            sys.exit(1)
        
        logger.info("Successfully logged in to ECR")
        return registry_url
    except Exception as e:
        logger.error(f"Docker login process failed: {e}")
        sys.exit(1)

def build_and_push_image(config, use_cache=True, skip_repo_check=False, local_build_only=False, max_retries=5):
    """Build and push Docker image to ECR for linux/amd64 platform."""
    region = get_aws_region(config)
    
    # Login to ECR first to get registry URL
    registry_url = login_to_ecr(region)
    
    # Extract repository name and tag from custom_image
    custom_image = config['container']['custom_image']
    if ':' in custom_image:
        repo_full_path, image_tag = custom_image.split(':')
        repo_name = repo_full_path.split('/')[-1]
    else:
        repo_name = custom_image.split('/')[-1]
        image_tag = 'latest'
    
    # Check if repository exists, create if not
    if not skip_repo_check:
        describe_cmd = ["aws", "ecr", "describe-repositories", "--repository-names", repo_name, "--region", region, "--output", "json"]
        try:
            run_command(describe_cmd, shell=False, check=False)
            logger.info(f"Repository {repo_name} exists")
        except subprocess.CalledProcessError:
            logger.info(f"Repository {repo_name} doesn't exist. Creating it...")
            create_cmd = ["aws", "ecr", "create-repository", "--repository-name", repo_name, "--region", region, "--output", "json"]
            run_command(create_cmd, shell=False)
    
    # Get full image name including registry
    full_image_name = f"{registry_url}/{repo_name}:{image_tag}"
    logger.info(f"Building image: {full_image_name}")
    
    # Navigate to the Dockerfile directory
    dockerfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 config['container']['dockerfile_dir'])
    os.chdir(dockerfile_dir)
    logger.info(f"Building Docker image from directory: {dockerfile_dir}")
    
    # Build for linux/amd64 platform
    logger.info("Building image for linux/amd64 platform")
    build_cmd = f"docker build -t {full_image_name} --platform linux/amd64"
    
    if not use_cache:
        build_cmd += " --no-cache"
        
    build_cmd += " ."
    
    # Execute build with real-time output
    run_command_realtime(build_cmd)
    
    # Print image size
    image_size_cmd = f"docker image ls {full_image_name} --format '{{{{.Size}}}}'"
    try:
        image_size = subprocess.check_output(image_size_cmd, shell=True, text=True).strip()
        logger.info(f"Built image size: {image_size}")
    except Exception as e:
        logger.warning(f"Could not determine image size: {str(e)}")
    
    if not local_build_only:
        # Push directly using docker push with real-time output
        logger.info(f"Pushing image {full_image_name} to ECR (showing progress)...")
        
        # Re-login to ECR before pushing to ensure the credentials are fresh
        login_to_ecr(region)
        
        # Push directly with real-time output
        push_cmd = f"docker push {full_image_name}"
        run_command_realtime(push_cmd)
        logger.info(f"Successfully pushed image to {full_image_name}")
    else:
        logger.info(f"Local build completed. To push, run: docker push {full_image_name}")
    
    return full_image_name

def main():
    parser = argparse.ArgumentParser(description='Build and push Docker image to ECR for linux/amd64')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml file')
    parser.add_argument('--tag', help='Override the image tag specified in config.yaml')
    parser.add_argument('--no-cache', action='store_true', help='Disable Docker build cache')
    parser.add_argument('--skip-repo-check', action='store_true', help='Skip repository check/creation')
    parser.add_argument('--local-build', action='store_true', help='Build locally only without pushing to ECR')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of push retry attempts')
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.abspath(args.config)
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Override tag if provided
    if args.tag:
        logger.info(f"Overriding image tag with: {args.tag}")
        repo_part = config['container']['custom_image'].split(':')[0]
        config['container']['custom_image'] = f"{repo_part}:{args.tag}"
    
    # Build and push image
    build_and_push_image(
        config, 
        use_cache=not args.no_cache, 
        skip_repo_check=args.skip_repo_check,
        local_build_only=args.local_build,
        max_retries=args.max_retries
    )

if __name__ == "__main__":
    main() 