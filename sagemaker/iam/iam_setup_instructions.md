# Setting up IAM Role for SageMaker

This document describes the steps for creating and configuring an IAM role for deploying machine learning models in Amazon SageMaker.

## Prerequisites

- AWS CLI installed and configured
- Necessary permissions to create and manage IAM roles

## Step-by-Step Instructions

### 1. Creating trust-policy.json

First, you need to create a `trust-policy.json` file that defines who can use the role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole",
      "Condition": {}
    }
  ]
}
```

### 2. Creating S3 Access Policy

Create a `sagemaker-s3-policy.json` file that defines S3 access permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": "*"
        }
    ]
}
```

### 3. Creating IAM Role

Execute the following command to create an IAM role with the required trust policy:

```bash
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document file://trust-policy.json \
    --description "Role for SageMaker to access AWS resources"
```

### 4. Attaching AWS Managed Policies

Attach standard AWS policies for working with SageMaker and S3:

```bash
# Attaching SageMaker full access policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Attaching S3 full access policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Attaching CloudWatch Logs access policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
```

### 5. Adding Custom Policies

Create a `sagemaker-cloudwatch-policy.json` file to configure access rights to CloudWatch Logs:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
                "logs:GetLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

Add custom inline policies:

```bash
# Adding S3 policy
aws iam put-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-name SageMakerS3BucketAccess \
    --policy-document file://sagemaker-s3-policy.json

# Adding CloudWatch Logs policy
aws iam put-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-name SageMakerCloudWatchLogsAccess \
    --policy-document file://sagemaker-cloudwatch-policy.json
```

### 6. Verifying the Created Role

To verify the created role and attached policies, execute:

```bash
# Viewing role information
aws iam get-role --role-name SageMakerExecutionRole

# Viewing attached policies
aws iam list-attached-role-policies --role-name SageMakerExecutionRole

# Viewing inline policies
aws iam list-role-policies --role-name SageMakerExecutionRole

# Viewing inline policy content
aws iam get-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-name SageMakerS3BucketAccess
```

### 7. Getting the Role ARN

Save the role ARN for use when deploying models:

```bash
aws iam get-role \
    --role-name SageMakerExecutionRole \
    --query 'Role.Arn' \
    --output text
```

The obtained role ARN (for example, `arn:aws:iam::123456789012:role/SageMakerExecutionRole`) should be specified in the `.env` or `config.yaml` file.

### 8. Updating the Trust Policy (if necessary)

If you need to update the trust policy:

```bash
aws iam update-assume-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-document file://trust-policy.json
```

## Removing Created Resources

If you need to remove the created resources:

```bash
# Removing inline policy
aws iam delete-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-name SageMakerS3BucketAccess

# Detaching attached policies
aws iam detach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam detach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Deleting the role
aws iam delete-role --role-name SageMakerExecutionRole
```

## Notes

- After making changes to policies, it may take some time for them to propagate.
- It is recommended to use the least privileged access, limiting permissions to only necessary actions and resources.
- For production environments, it is recommended to create more restricted and specific S3 access policies. 