"""
Configuration module for the Joke CLI application.

Contains constants, default settings, and configuration management.
"""

import os
from pathlib import Path
from typing import List

# AWS Bedrock Configuration
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_AWS_REGION = "us-east-1"
MAX_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# API Configuration
API_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3

# Joke Categories
AVAILABLE_CATEGORIES = [
    "general",
    "programming", 
    "dad-jokes",
    "puns",
    "clean"
]

# Feedback Configuration
FEEDBACK_RATING_MIN = 1
FEEDBACK_RATING_MAX = 5
FEEDBACK_STORAGE_FILENAME = "joke_feedback.json"

# File Paths
HOME_DIR = Path.home()
FEEDBACK_STORAGE_PATH = HOME_DIR / ".joke_cli" / FEEDBACK_STORAGE_FILENAME

# CLI Configuration
CLI_COMMAND_NAME = "joke"
CLI_DESCRIPTION = "Generate jokes using AWS Bedrock AI models"

# Exit Codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGUMENTS = 2
EXIT_AWS_CREDENTIALS_ERROR = 3
EXIT_AWS_ACCESS_DENIED = 4
EXIT_NETWORK_ERROR = 5
EXIT_SERVICE_UNAVAILABLE = 6
EXIT_RATE_LIMITED = 7
EXIT_USER_CANCELLED = 130  # Standard for SIGINT (Ctrl+C)

# Error Messages with actionable guidance
ERROR_MESSAGES = {
    "no_credentials": {
        "message": "AWS credentials not found.",
        "guidance": [
            "Configure credentials using one of these methods:",
            "  1. Run 'aws configure' to set up default credentials",
            "  2. Set AWS_PROFILE environment variable to use a specific profile",
            "  3. Use IAM roles if running on EC2/ECS/Lambda",
            "  4. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        ],
        "exit_code": EXIT_AWS_CREDENTIALS_ERROR
    },
    "invalid_profile": {
        "message": "AWS profile '{profile}' not found.",
        "guidance": [
            "Check available profiles:",
            "  1. Run 'aws configure list-profiles' to see available profiles",
            "  2. Verify the profile name is spelled correctly",
            "  3. Create the profile using 'aws configure --profile {profile}'"
        ],
        "exit_code": EXIT_AWS_CREDENTIALS_ERROR
    },
    "access_denied": {
        "message": "Access denied to Bedrock model '{model_id}'.",
        "guidance": [
            "Request access to the model:",
            "  1. Go to AWS Bedrock console (https://console.aws.amazon.com/bedrock/)",
            "  2. Navigate to 'Model access' in the left sidebar",
            "  3. Request access to '{model_id}' model",
            "  4. Wait for approval (usually takes a few minutes)",
            "  5. Ensure your IAM user/role has 'bedrock:InvokeModel' permission"
        ],
        "exit_code": EXIT_AWS_ACCESS_DENIED
    },
    "insufficient_permissions": {
        "message": "Insufficient permissions for AWS Bedrock operations.",
        "guidance": [
            "Required IAM permissions:",
            "  - bedrock:InvokeModel",
            "  - bedrock:ListFoundationModels (optional, for model listing)",
            "Contact your AWS administrator to add these permissions to your IAM user or role."
        ],
        "exit_code": EXIT_AWS_ACCESS_DENIED
    },
    "invalid_category": {
        "message": "Invalid joke category '{category}'.",
        "guidance": [
            "Available categories: {available_categories}",
            "Use --help to see all available options."
        ],
        "exit_code": EXIT_INVALID_ARGUMENTS
    },
    "network_error": {
        "message": "Network connection error occurred.",
        "guidance": [
            "Troubleshooting steps:",
            "  1. Check your internet connection",
            "  2. Verify AWS service status at https://status.aws.amazon.com/",
            "  3. Check if you're behind a corporate firewall",
            "  4. Try again in a few moments"
        ],
        "exit_code": EXIT_NETWORK_ERROR
    },
    "service_unavailable": {
        "message": "AWS Bedrock service is currently unavailable.",
        "guidance": [
            "This is usually temporary:",
            "  1. Check AWS service status at https://status.aws.amazon.com/",
            "  2. Wait a few minutes and try again",
            "  3. Try a different AWS region if the issue persists"
        ],
        "exit_code": EXIT_SERVICE_UNAVAILABLE
    },
    "rate_limit": {
        "message": "Rate limit exceeded for AWS Bedrock API.",
        "guidance": [
            "You're making requests too quickly:",
            "  1. Wait 30-60 seconds before trying again",
            "  2. Consider requesting higher rate limits in AWS console",
            "  3. Implement exponential backoff in automated scripts"
        ],
        "exit_code": EXIT_RATE_LIMITED
    },
    "timeout_error": {
        "message": "Request timed out after {timeout} seconds.",
        "guidance": [
            "The request took too long to complete:",
            "  1. Check your network connection stability",
            "  2. Try again - this may be a temporary issue",
            "  3. Consider using a different AWS region if problems persist"
        ],
        "exit_code": EXIT_NETWORK_ERROR
    },
    "model_not_found": {
        "message": "Bedrock model '{model_id}' not found or not available.",
        "guidance": [
            "Model availability issues:",
            "  1. Verify the model ID is correct",
            "  2. Check if the model is available in your AWS region",
            "  3. Ensure you have requested access to this model",
            "  4. Try using the default model: {default_model}"
        ],
        "exit_code": EXIT_GENERAL_ERROR
    },
    "empty_response": {
        "message": "The AI model returned an empty response.",
        "guidance": [
            "This can happen occasionally:",
            "  1. Try generating another joke",
            "  2. Try a different category",
            "  3. If this persists, there may be an issue with the model"
        ],
        "exit_code": EXIT_GENERAL_ERROR
    },
    "feedback_storage_error": {
        "message": "Failed to save feedback data.",
        "guidance": [
            "Feedback storage issues:",
            "  1. Check if you have write permissions to your home directory",
            "  2. Ensure sufficient disk space is available",
            "  3. The joke was still generated successfully"
        ],
        "exit_code": EXIT_GENERAL_ERROR
    },
    "invalid_rating": {
        "message": "Invalid rating '{rating}'. Rating must be between 1 and 5.",
        "guidance": [
            "Please provide a valid rating:",
            "  1 = Poor",
            "  2 = Fair", 
            "  3 = Good",
            "  4 = Very Good",
            "  5 = Excellent",
            "Or enter 's' to skip feedback."
        ],
        "exit_code": EXIT_INVALID_ARGUMENTS
    },
    "general_error": {
        "message": "An unexpected error occurred: {error}",
        "guidance": [
            "This is an unexpected error:",
            "  1. Try running the command again",
            "  2. Check that your AWS credentials are properly configured",
            "  3. If the problem persists, please report this issue"
        ],
        "exit_code": EXIT_GENERAL_ERROR
    }
}

def get_feedback_storage_dir() -> Path:
    """Get the directory for storing feedback data."""
    storage_dir = HOME_DIR / ".joke_cli"
    storage_dir.mkdir(exist_ok=True)
    return storage_dir

def get_aws_profile() -> str:
    """Get AWS profile from environment variable or return None."""
    return os.environ.get("AWS_PROFILE")

def get_aws_region() -> str:
    """Get AWS region from environment variable or return default."""
    return os.environ.get("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION)