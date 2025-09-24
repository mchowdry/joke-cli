"""
Unit tests for the bedrock_client module.

Tests AWS Bedrock client integration with mocked responses for both
success and failure scenarios.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import (
    ClientError, 
    NoCredentialsError, 
    PartialCredentialsError,
    ConnectionError,
    ReadTimeoutError
)

from joke_cli.bedrock_client import (
    BedrockClient, 
    BedrockClientError, 
    create_bedrock_client, 
    invoke_model
)
from joke_cli.models import BedrockConfig
from joke_cli.config import DEFAULT_AWS_REGION, API_TIMEOUT_SECONDS, MAX_RETRIES


class TestBedrockClient:
    """Test cases for BedrockClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = BedrockClient()
        self.config = BedrockConfig(model_id="amazon.titan-text-express-v1")
        self.sample_prompt = "Tell me a joke about programming"
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_create_session_without_profile(self, mock_session_class):
        """Test creating session without AWS profile."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = BedrockClient()
        session = client._create_session()
        
        mock_session_class.assert_called_once_with()
        assert session == mock_session
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_create_session_with_profile(self, mock_session_class):
        """Test creating session with AWS profile."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = BedrockClient(profile="test-profile")
        session = client._create_session()
        
        mock_session_class.assert_called_once_with(profile_name="test-profile")
        assert session == mock_session
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_get_client_success(self, mock_session_class):
        """Test successful client creation."""
        mock_session = Mock()
        mock_bedrock_client = Mock()
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # Mock successful credentials test
        mock_bedrock_client.list_foundation_models.return_value = {}
        
        client = BedrockClient()
        bedrock_client = client._get_client()
        
        assert bedrock_client == mock_bedrock_client
        mock_bedrock_client.list_foundation_models.assert_called_once()
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_get_client_no_credentials(self, mock_session_class):
        """Test client creation with no credentials."""
        mock_session = Mock()
        mock_session.client.side_effect = NoCredentialsError()
        mock_session_class.return_value = mock_session
        
        client = BedrockClient()
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._get_client()
        
        assert "AWS credentials not found" in str(exc_info.value)
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_get_client_partial_credentials(self, mock_session_class):
        """Test client creation with partial credentials."""
        mock_session = Mock()
        mock_session.client.side_effect = PartialCredentialsError(
            provider="test", cred_var="AWS_ACCESS_KEY_ID"
        )
        mock_session_class.return_value = mock_session
        
        client = BedrockClient()
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._get_client()
        
        assert "AWS credentials not found" in str(exc_info.value)
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_test_credentials_access_denied(self, mock_session_class):
        """Test credentials test with access denied."""
        mock_session = Mock()
        mock_bedrock_client = Mock()
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # Mock access denied error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_bedrock_client.list_foundation_models.side_effect = ClientError(
            error_response, 'ListFoundationModels'
        )
        
        client = BedrockClient()
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._get_client()
        
        assert "Access denied to Bedrock model" in str(exc_info.value)
    
    def test_invoke_model_titan_success(self):
        """Test successful model invocation with Titan model."""
        mock_client = Mock()
        
        # Mock successful response
        response_body = {
            'results': [{'outputText': 'Why do programmers prefer dark mode? Because light attracts bugs!'}]
        }
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        result = client.invoke_model(self.sample_prompt, self.config)
        
        assert result == "Why do programmers prefer dark mode? Because light attracts bugs!"
        mock_client.invoke_model.assert_called_once()
        
        # Verify request body structure
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]['body'])
        assert 'inputText' in body
        assert 'textGenerationConfig' in body
        assert body['inputText'] == self.sample_prompt
    
    def test_invoke_model_claude_success(self):
        """Test successful model invocation with Claude model."""
        mock_client = Mock()
        
        # Mock successful response
        response_body = {
            'completion': 'Why do programmers prefer dark mode? Because light attracts bugs!'
        }
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        claude_config = BedrockConfig(model_id="anthropic.claude-v2")
        result = client.invoke_model(self.sample_prompt, claude_config)
        
        assert result == "Why do programmers prefer dark mode? Because light attracts bugs!"
        
        # Verify Claude-specific request format
        call_args = mock_client.invoke_model.call_args
        body = json.loads(call_args[1]['body'])
        assert 'prompt' in body
        assert 'max_tokens_to_sample' in body
        assert "Human:" in body['prompt']
        assert "Assistant:" in body['prompt']
    
    def test_invoke_model_empty_response(self):
        """Test model invocation with empty response."""
        mock_client = Mock()
        
        # Mock empty response
        response_body = {'results': [{'outputText': ''}]}
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Model returned empty response" in str(exc_info.value)
    
    def test_invoke_model_access_denied(self):
        """Test model invocation with access denied error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}}
        mock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Access denied to Bedrock model" in str(exc_info.value)
    
    def test_invoke_model_throttling(self):
        """Test model invocation with throttling error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
        mock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_invoke_model_service_unavailable(self):
        """Test model invocation with service unavailable error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'ServiceUnavailableException', 'Message': 'Service unavailable'}}
        mock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "AWS Bedrock service is currently unavailable" in str(exc_info.value)
    
    def test_invoke_model_validation_error(self):
        """Test model invocation with validation error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Invalid input'}}
        mock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Invalid request: Invalid input" in str(exc_info.value)
    
    def test_invoke_model_resource_not_found(self):
        """Test model invocation with resource not found error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Model not found'}}
        mock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "not found or not available" in str(exc_info.value)
    
    def test_invoke_model_connection_error(self):
        """Test model invocation with connection error."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ConnectionError(error="Connection failed")
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Network connection error occurred" in str(exc_info.value)
    
    def test_invoke_model_timeout_error(self):
        """Test model invocation with timeout error."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ReadTimeoutError(endpoint_url="test")
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Request timed out after" in str(exc_info.value)
    
    def test_invoke_model_json_decode_error(self):
        """Test model invocation with invalid JSON response."""
        mock_client = Mock()
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = b"invalid json"
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        assert "Invalid response format" in str(exc_info.value)
    
    def test_test_connection_success(self):
        """Test successful connection test."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {}
        
        client = BedrockClient()
        client._client = mock_client
        
        result = client.test_connection()
        
        assert result is True
        mock_client.list_foundation_models.assert_called_once()
    
    def test_test_connection_failure(self):
        """Test failed connection test."""
        mock_client = Mock()
        mock_client.list_foundation_models.side_effect = Exception("Connection failed")
        
        client = BedrockClient()
        client._client = mock_client
        
        result = client.test_connection()
        
        assert result is False
    
    def test_list_available_models_success(self):
        """Test successful model listing."""
        mock_client = Mock()
        
        mock_response = {
            'modelSummaries': [
                {
                    'modelId': 'amazon.titan-text-express-v1',
                    'modelName': 'Titan Text Express',
                    'providerName': 'Amazon',
                    'inputModalities': ['TEXT'],
                    'outputModalities': ['TEXT']
                },
                {
                    'modelId': 'anthropic.claude-v2',
                    'modelName': 'Claude v2',
                    'providerName': 'Anthropic',
                    'inputModalities': ['TEXT'],
                    'outputModalities': ['TEXT']
                }
            ]
        }
        mock_client.list_foundation_models.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        models = client.list_available_models()
        
        assert len(models) == 2
        assert models[0]['modelId'] == 'amazon.titan-text-express-v1'
        assert models[1]['modelId'] == 'anthropic.claude-v2'
    
    def test_list_available_models_error(self):
        """Test model listing with error."""
        mock_client = Mock()
        
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_client.list_foundation_models.side_effect = ClientError(
            error_response, 'ListFoundationModels'
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError):
            client.list_available_models()


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_bedrock_client(self):
        """Test bedrock client factory function."""
        client = create_bedrock_client()
        
        assert isinstance(client, BedrockClient)
        assert client.profile is None
        assert client.region == "us-east-1"  # Default region
    
    def test_create_bedrock_client_with_params(self):
        """Test bedrock client factory function with parameters."""
        client = create_bedrock_client(profile="test-profile", region="us-west-2")
        
        assert isinstance(client, BedrockClient)
        assert client.profile == "test-profile"
        assert client.region == "us-west-2"
    
    @patch('joke_cli.bedrock_client.create_bedrock_client')
    def test_invoke_model_convenience_function(self, mock_create_client):
        """Test convenience invoke_model function."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Test joke"
        mock_create_client.return_value = mock_client
        
        result = invoke_model("Test prompt", "test-model")
        
        assert result == "Test joke"
        mock_create_client.assert_called_once_with(profile=None)
        mock_client.invoke_model.assert_called_once()
    
    @patch('joke_cli.bedrock_client.create_bedrock_client')
    def test_invoke_model_convenience_function_with_config(self, mock_create_client):
        """Test convenience invoke_model function with custom config."""
        mock_client = Mock()
        mock_client.invoke_model.return_value = "Test joke"
        mock_create_client.return_value = mock_client
        
        config = BedrockConfig(model_id="test-model", temperature=0.5)
        result = invoke_model("Test prompt", "test-model", config=config)
        
        assert result == "Test joke"
        mock_client.invoke_model.assert_called_once()
        
        # Verify the config was passed correctly
        call_args = mock_client.invoke_model.call_args
        passed_config = call_args[0][1]  # Second argument is the config
        assert passed_config.temperature == 0.5


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_prompt = "Tell me a joke about programming"
        self.config = BedrockConfig(model_id="amazon.titan-text-express-v1")
    
    def test_handle_client_error_unknown_code(self):
        """Test handling of unknown error codes."""
        client = BedrockClient()
        
        error_response = {'Error': {'Code': 'UnknownError', 'Message': 'Unknown error occurred'}}
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "test-model")
        
        assert "AWS API error (UnknownError)" in str(exc_info.value)
        assert "Unknown error occurred" in str(exc_info.value)
    
    def test_handle_client_error_missing_error_info(self):
        """Test handling of client error with missing error information."""
        client = BedrockClient()
        
        error_response = {}  # Missing Error key
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "test-model")
        
        assert "AWS API error (Unknown)" in str(exc_info.value)
    
    def test_handle_client_error_access_denied_enhanced(self):
        """Test enhanced error handling for access denied errors."""
        client = BedrockClient()
        
        error_response = {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}}
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "amazon.titan-text-express-v1")
        
        # Should use enhanced error message from error handler
        assert "Access denied to Bedrock model 'amazon.titan-text-express-v1'" in str(exc_info.value)
    
    def test_handle_client_error_rate_limit_enhanced(self):
        """Test enhanced error handling for rate limit errors."""
        client = BedrockClient()
        
        error_response = {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}}
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "test-model")
        
        # Should use enhanced error message from error handler
        assert "Rate limit exceeded for AWS Bedrock API" in str(exc_info.value)
    
    def test_handle_client_error_service_unavailable_enhanced(self):
        """Test enhanced error handling for service unavailable errors."""
        client = BedrockClient()
        
        error_response = {'Error': {'Code': 'ServiceUnavailableException', 'Message': 'Service unavailable'}}
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "test-model")
        
        # Should use enhanced error message from error handler
        assert "AWS Bedrock service is currently unavailable" in str(exc_info.value)
    
    def test_handle_client_error_model_not_found_enhanced(self):
        """Test enhanced error handling for model not found errors."""
        client = BedrockClient()
        
        error_response = {'Error': {'Code': 'ResourceNotFoundException', 'Message': 'Model not found'}}
        error = ClientError(error_response, 'InvokeModel')
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._handle_client_error(error, "invalid-model")
        
        # Should use enhanced error message from error handler
        assert "Bedrock model 'invalid-model' not found or not available" in str(exc_info.value)
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_test_credentials_invalid_profile(self, mock_session_class):
        """Test credentials test with invalid profile error."""
        mock_session = Mock()
        mock_bedrock_client = Mock()
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # Mock invalid profile error
        error_response = {'Error': {'Code': 'InvalidUserID.NotFound', 'Message': 'Profile not found'}}
        mock_bedrock_client.list_foundation_models.side_effect = ClientError(
            error_response, 'ListFoundationModels'
        )
        
        client = BedrockClient(profile="invalid-profile")
        
        with pytest.raises(BedrockClientError) as exc_info:
            client._get_client()
        
        assert "AWS profile 'invalid-profile' not found" in str(exc_info.value)
    
    def test_invoke_model_timeout_specific_error(self):
        """Test model invocation with specific timeout error handling."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ReadTimeoutError(
            endpoint_url="test", 
            error="Read timeout on endpoint URL"
        )
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        # Should detect timeout in error message
        error_msg = str(exc_info.value)
        assert "Request timed out after" in error_msg or "Network error occurred" in error_msg
    
    def test_invoke_model_empty_response_enhanced(self):
        """Test enhanced error handling for empty model responses."""
        mock_client = Mock()
        
        # Mock empty response
        response_body = {'results': [{'outputText': '   '}]}  # Only whitespace
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(response_body).encode()
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            client.invoke_model(self.sample_prompt, self.config)
        
        # Should use enhanced error message (could be either specific or general error)
        error_msg = str(exc_info.value)
        assert ("AI model returned an empty response" in error_msg or 
                "Model returned empty response" in error_msg)


class TestEnhancedErrorMessages:
    """Test cases for enhanced error message formatting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = BedrockClient()
        self.config = BedrockConfig(model_id="amazon.titan-text-express-v1")
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_no_credentials_error_message(self, mock_session_class):
        """Test enhanced no credentials error message."""
        mock_session = Mock()
        mock_session.client.side_effect = NoCredentialsError()
        mock_session_class.return_value = mock_session
        
        with pytest.raises(BedrockClientError) as exc_info:
            self.client._get_client()
        
        error_msg = str(exc_info.value)
        assert "AWS credentials not found" in error_msg
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_access_denied_error_message(self, mock_session_class):
        """Test enhanced access denied error message."""
        mock_session = Mock()
        mock_bedrock_client = Mock()
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}}
        mock_bedrock_client.list_foundation_models.side_effect = ClientError(
            error_response, 'ListFoundationModels'
        )
        
        with pytest.raises(BedrockClientError) as exc_info:
            self.client._get_client()
        
        error_msg = str(exc_info.value)
        assert "Access denied to Bedrock model" in error_msg
    
    def test_network_error_message(self):
        """Test enhanced network error message."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = ConnectionError(error="Network unreachable")
        
        self.client._client = mock_client
        
        with pytest.raises(BedrockClientError) as exc_info:
            self.client.invoke_model("test prompt", self.config)
        
        error_msg = str(exc_info.value)
        assert "Network connection error occurred" in error_msg or "Network error occurred" in error_msg


if __name__ == '__main__':
    pytest.main([__file__])