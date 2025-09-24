"""
AWS Bedrock client integration for the Joke CLI application.

This module handles all interactions with AWS Bedrock API, including
client initialization, model invocation, and comprehensive error handling.
"""

import json
import logging
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import (
    ClientError, 
    NoCredentialsError, 
    PartialCredentialsError,
    BotoCoreError,
    ConnectionError,
    ReadTimeoutError
)
from botocore.config import Config

from .models import BedrockConfig, JokeResponse
from .config import (
    DEFAULT_AWS_REGION,
    API_TIMEOUT_SECONDS,
    MAX_RETRIES,
    DEFAULT_MODEL_ID
)
from .error_handler import get_error_handler


logger = logging.getLogger(__name__)


class BedrockClientError(Exception):
    """Custom exception for Bedrock client errors."""
    pass


class BedrockClient:
    """AWS Bedrock client for joke generation."""
    
    def __init__(self, profile: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize the Bedrock client.
        
        Args:
            profile: AWS profile name to use
            region: AWS region to use (defaults to DEFAULT_AWS_REGION)
        """
        self.profile = profile
        self.region = region or DEFAULT_AWS_REGION
        self._client = None
        
    def _create_session(self) -> boto3.Session:
        """Create a boto3 session with optional profile."""
        if self.profile:
            return boto3.Session(profile_name=self.profile)
        return boto3.Session()
    
    def _get_client(self):
        """Get or create the Bedrock client with proper configuration."""
        if self._client is None:
            try:
                session = self._create_session()
                
                # Configure client with timeouts and retries
                config = Config(
                    region_name=self.region,
                    retries={
                        'max_attempts': MAX_RETRIES,
                        'mode': 'adaptive'
                    },
                    read_timeout=API_TIMEOUT_SECONDS,
                    connect_timeout=API_TIMEOUT_SECONDS
                )
                
                self._client = session.client('bedrock-runtime', config=config)
                
                # Test credentials by making a simple call
                self._test_credentials()
                
            except (NoCredentialsError, PartialCredentialsError) as e:
                logger.error(f"AWS credentials error: {e}")
                error_handler = get_error_handler()
                error_info = error_handler.format_error_message("no_credentials")
                raise BedrockClientError(error_info["message"]) from e
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_handler = get_error_handler()
                if error_code == 'UnauthorizedOperation':
                    error_info = error_handler.format_error_message("access_denied", model_id="bedrock-runtime")
                    raise BedrockClientError(error_info["message"]) from e
                raise BedrockClientError(f"AWS client error ({error_code}): {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error creating Bedrock client: {e}")
                raise BedrockClientError(f"Failed to initialize Bedrock client: {e}") from e
                
        return self._client
    
    def _test_credentials(self) -> None:
        """Test if credentials are valid by making a simple API call."""
        try:
            # Check if this is a newer Claude model that requires Converse API
            if "claude" in DEFAULT_MODEL_ID.lower() and ("claude-3" in DEFAULT_MODEL_ID.lower() or "claude-sonnet-4" in DEFAULT_MODEL_ID.lower()):
                # Use Converse API for newer Claude models
                try:
                    self._client.converse(
                        modelId=DEFAULT_MODEL_ID,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "test"
                                    }
                                ]
                            }
                        ],
                        inferenceConfig={
                            "maxTokens": 1,
                            "temperature": 0.1,
                            "topP": 0.9
                        }
                    )
                except ClientError as invoke_error:
                    # If it's an access denied error, re-raise it
                    # If it's a validation error, that's actually fine - it means credentials work
                    error_code = invoke_error.response.get('Error', {}).get('Code', 'Unknown')
                    if error_code in ['AccessDeniedException', 'UnauthorizedOperation']:
                        raise invoke_error
                    # Other errors like ValidationException are acceptable for credential testing
            else:
                # Use legacy invoke_model API for older models
                if "claude" in DEFAULT_MODEL_ID.lower():
                    # Legacy Claude format
                    test_body = json.dumps({
                        "prompt": "\n\nHuman: test\n\nAssistant:",
                        "max_tokens_to_sample": 1,
                        "temperature": 0.1,
                        "top_p": 0.9
                    })
                else:
                    # Fallback to Titan format
                    test_body = json.dumps({
                        "inputText": "test",
                        "textGenerationConfig": {
                            "maxTokenCount": 1,
                            "temperature": 0.1
                        }
                    })
                
                try:
                    self._client.invoke_model(
                        modelId=DEFAULT_MODEL_ID,
                        body=test_body,
                        contentType='application/json',
                        accept='application/json'
                    )
                except ClientError as invoke_error:
                    # If it's an access denied error, re-raise it
                    # If it's a validation error, that's actually fine - it means credentials work
                    error_code = invoke_error.response.get('Error', {}).get('Code', 'Unknown')
                    if error_code in ['AccessDeniedException', 'UnauthorizedOperation']:
                        raise invoke_error
                    # Other errors like ValidationException are acceptable for credential testing
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_handler = get_error_handler()
            if error_code in ['AccessDeniedException', 'UnauthorizedOperation']:
                error_info = error_handler.format_error_message("access_denied", model_id="bedrock-runtime")
                raise BedrockClientError(error_info["message"]) from e
            elif error_code == 'InvalidUserID.NotFound':
                error_info = error_handler.format_error_message("invalid_profile", profile=self.profile or "default")
                raise BedrockClientError(error_info["message"]) from e
            # Re-raise other client errors
            raise
    
    def invoke_model(self, prompt: str, config: BedrockConfig) -> str:
        """
        Invoke a Bedrock model with the given prompt and configuration.
        
        Args:
            prompt: The prompt to send to the model
            config: Bedrock configuration including model ID and parameters
            
        Returns:
            The generated text response from the model
            
        Raises:
            BedrockClientError: If the API call fails
        """
        try:
            client = self._get_client()
            
            # Check if this is a newer Claude model that requires Converse API
            if "claude" in config.model_id.lower() and ("claude-3" in config.model_id.lower() or "claude-sonnet-4" in config.model_id.lower()):
                return self._invoke_with_converse_api(client, prompt, config)
            else:
                return self._invoke_with_legacy_api(client, prompt, config)
            
        except ClientError as e:
            return self._handle_client_error(e, config.model_id)
        except (ConnectionError, ReadTimeoutError) as e:
            logger.error(f"Network error: {e}")
            error_handler = get_error_handler()
            if "timeout" in str(e).lower():
                error_info = error_handler.format_error_message("timeout_error", timeout=API_TIMEOUT_SECONDS)
                raise BedrockClientError(error_info["message"]) from e
            else:
                error_info = error_handler.format_error_message("network_error")
                raise BedrockClientError(error_info["message"]) from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response JSON: {e}")
            raise BedrockClientError("Invalid response format from Bedrock API") from e
        except BotoCoreError as e:
            logger.error(f"Boto core error: {e}")
            raise BedrockClientError(f"AWS SDK error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during model invocation: {e}")
            raise BedrockClientError(f"Unexpected error: {e}") from e
    
    def _invoke_with_converse_api(self, client, prompt: str, config: BedrockConfig) -> str:
        """Use the Converse API for newer Claude models."""
        logger.debug(f"Using Converse API for model {config.model_id}")
        
        response = client.converse(
            modelId=config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            inferenceConfig={
                "maxTokens": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p
            }
        )
        
        # Extract text from Converse API response
        output = response.get('output', {})
        message = output.get('message', {})
        content = message.get('content', [])
        
        if content and len(content) > 0:
            generated_text = content[0].get('text', '')
        else:
            raise BedrockClientError("Model returned empty response")
        
        if not generated_text or not generated_text.strip():
            raise BedrockClientError("Model returned empty response")
        
        logger.debug(f"Successfully generated text of length: {len(generated_text)}")
        return generated_text.strip()
    
    def _invoke_with_legacy_api(self, client, prompt: str, config: BedrockConfig) -> str:
        """Use the legacy invoke_model API for older models."""
        logger.debug(f"Using legacy API for model {config.model_id}")
        
        # Prepare the request body based on the model
        if "titan" in config.model_id.lower():
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": config.max_tokens,
                    "temperature": config.temperature,
                    "topP": config.top_p,
                    "stopSequences": []
                }
            }
        elif "claude" in config.model_id.lower():
            # Use legacy format for older Claude models
            request_body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p
            }
        else:
            # Generic format - may need adjustment for specific models
            request_body = {
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p
            }
        
        response = client.invoke_model(
            modelId=config.model_id,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract text based on model type
        if "titan" in config.model_id.lower():
            generated_text = response_body.get('results', [{}])[0].get('outputText', '')
        elif "claude" in config.model_id.lower():
            generated_text = response_body.get('completion', '')
        else:
            # Try common response fields
            generated_text = (
                response_body.get('generated_text') or 
                response_body.get('text') or 
                response_body.get('output') or
                str(response_body)
            )
        
        if not generated_text or not generated_text.strip():
            raise BedrockClientError("Model returned empty response")
        
        logger.debug(f"Successfully generated text of length: {len(generated_text)}")
        return generated_text.strip()
    
    def _handle_client_error(self, error: ClientError, model_id: str) -> None:
        """
        Handle AWS ClientError exceptions with specific error messages.
        
        Args:
            error: The ClientError exception
            model_id: The model ID that was being invoked
            
        Raises:
            BedrockClientError: With appropriate error message
        """
        error_code = error.response.get('Error', {}).get('Code', 'Unknown')
        error_message = error.response.get('Error', {}).get('Message', str(error))
        
        logger.error(f"AWS ClientError - Code: {error_code}, Message: {error_message}")
        
        error_handler = get_error_handler()
        
        if error_code == 'AccessDeniedException':
            error_info = error_handler.format_error_message("access_denied", model_id=model_id)
            raise BedrockClientError(error_info["message"]) from error
        elif error_code == 'ThrottlingException':
            error_info = error_handler.format_error_message("rate_limit")
            raise BedrockClientError(error_info["message"]) from error
        elif error_code == 'ServiceUnavailableException':
            error_info = error_handler.format_error_message("service_unavailable")
            raise BedrockClientError(error_info["message"]) from error
        elif error_code == 'ValidationException':
            raise BedrockClientError(f"Invalid request: {error_message}") from error
        elif error_code == 'ResourceNotFoundException':
            error_info = error_handler.format_error_message("model_not_found", model_id=model_id, default_model=DEFAULT_MODEL_ID)
            raise BedrockClientError(error_info["message"]) from error
        else:
            raise BedrockClientError(f"AWS API error ({error_code}): {error_message}") from error
    
    def test_connection(self) -> bool:
        """
        Test the connection to AWS Bedrock.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self._get_client()  # This will test credentials via _test_credentials
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def list_available_models(self) -> list:
        """
        List available foundation models.
        
        Returns:
            List of available model IDs
            
        Raises:
            BedrockClientError: If the API call fails
        """
        try:
            # Use bedrock client (not bedrock-runtime) for management operations
            session = self._create_session()
            config = Config(
                region_name=self.region,
                retries={
                    'max_attempts': MAX_RETRIES,
                    'mode': 'adaptive'
                },
                read_timeout=API_TIMEOUT_SECONDS,
                connect_timeout=API_TIMEOUT_SECONDS
            )
            
            bedrock_client = session.client('bedrock', config=config)
            response = bedrock_client.list_foundation_models()
            
            models = []
            for model in response.get('modelSummaries', []):
                models.append({
                    'modelId': model.get('modelId'),
                    'modelName': model.get('modelName'),
                    'providerName': model.get('providerName'),
                    'inputModalities': model.get('inputModalities', []),
                    'outputModalities': model.get('outputModalities', [])
                })
            
            return models
            
        except ClientError as e:
            self._handle_client_error(e, "list_models")
        except Exception as e:
            logger.error(f"Unexpected error listing models: {e}")
            raise BedrockClientError(f"Failed to list models: {e}") from e


def create_bedrock_client(profile: Optional[str] = None, region: Optional[str] = None) -> BedrockClient:
    """
    Factory function to create a Bedrock client.
    
    Args:
        profile: AWS profile name to use
        region: AWS region to use
        
    Returns:
        Configured BedrockClient instance
    """
    return BedrockClient(profile=profile, region=region)


def invoke_model(prompt: str, model_id: str, profile: Optional[str] = None, 
                config: Optional[BedrockConfig] = None) -> str:
    """
    Convenience function to invoke a model with a prompt.
    
    Args:
        prompt: The prompt to send to the model
        model_id: The model ID to invoke
        profile: AWS profile name to use
        config: Bedrock configuration (uses defaults if not provided)
        
    Returns:
        The generated text response
        
    Raises:
        BedrockClientError: If the API call fails
    """
    if config is None:
        config = BedrockConfig(model_id=model_id)
    
    client = create_bedrock_client(profile=profile)
    return client.invoke_model(prompt, config)