"""
Unit tests for the error handling module.

Tests error message formatting, logging configuration, and error handling workflows.
"""

import pytest
import sys
import logging
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path

from joke_cli.error_handler import (
    ErrorHandler,
    JokeCliError,
    get_error_handler,
    handle_error,
    display_error_message,
    validate_condition
)
from joke_cli.config import ERROR_MESSAGES, EXIT_SUCCESS, EXIT_GENERAL_ERROR


class TestJokeCliError:
    """Test the custom JokeCliError exception class."""
    
    def test_joke_cli_error_creation(self):
        """Test creating a JokeCliError with all parameters."""
        error = JokeCliError(
            message="Test error",
            error_code="test_error",
            exit_code=42,
            guidance=["Step 1", "Step 2"]
        )
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "test_error"
        assert error.exit_code == 42
        assert error.guidance == ["Step 1", "Step 2"]
    
    def test_joke_cli_error_defaults(self):
        """Test JokeCliError with default parameters."""
        error = JokeCliError("Test error")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "general_error"
        assert error.exit_code == 1
        assert error.guidance == []


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler(logger_name="test_logger", debug=False)
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler(logger_name="test", debug=True)
        
        assert handler.logger.name == "test"
        assert handler.debug is True
        assert handler.logger.level == logging.DEBUG
    
    def test_format_error_message_valid_code(self):
        """Test formatting a valid error message."""
        result = self.error_handler.format_error_message(
            "invalid_category",
            category="invalid",
            available_categories="general, programming"
        )
        
        assert "Invalid joke category 'invalid'" in result["message"]
        assert "Available categories: general, programming" in result["guidance"][0]
        assert result["exit_code"] == ERROR_MESSAGES["invalid_category"]["exit_code"]
    
    def test_format_error_message_invalid_code(self):
        """Test formatting an invalid error code."""
        result = self.error_handler.format_error_message("nonexistent_error")
        
        assert "Unknown error: nonexistent_error" in result["message"]
        assert "Please report this issue" in result["guidance"][0]
        assert result["exit_code"] == 1
    
    def test_format_error_message_missing_parameters(self):
        """Test formatting error message with missing parameters."""
        result = self.error_handler.format_error_message("invalid_category")
        
        # Should not crash and should return the unformatted message
        assert "Invalid joke category" in result["message"]
        assert len(result["guidance"]) > 0
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_display_error(self, mock_stderr):
        """Test displaying error messages."""
        self.error_handler.display_error(
            "Test error message",
            ["Step 1", "Step 2", "Step 3"]
        )
        
        output = mock_stderr.getvalue()
        assert "❌ Error: Test error message" in output
        assert "💡 How to fix this:" in output
        assert "Step 1" in output
        assert "Step 2" in output
        assert "Step 3" in output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_display_warning(self, mock_stderr):
        """Test displaying warning messages."""
        self.error_handler.display_warning("Test warning")
        
        output = mock_stderr.getvalue()
        assert "⚠️  Warning: Test warning" in output
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_info(self, mock_stdout):
        """Test displaying info messages."""
        self.error_handler.display_info("Test info")
        
        output = mock_stdout.getvalue()
        assert "ℹ️  Test info" in output
    
    def test_determine_error_code_credentials(self):
        """Test error code determination for credentials errors."""
        error = Exception("No credentials found")
        code = self.error_handler._determine_error_code(error)
        assert code == "no_credentials"
    
    def test_determine_error_code_access_denied(self):
        """Test error code determination for access denied errors."""
        error = Exception("Access denied to resource")
        code = self.error_handler._determine_error_code(error)
        assert code == "access_denied"
    
    def test_determine_error_code_rate_limit(self):
        """Test error code determination for rate limit errors."""
        error = Exception("Rate limit exceeded")
        code = self.error_handler._determine_error_code(error)
        assert code == "rate_limit"
    
    def test_determine_error_code_network(self):
        """Test error code determination for network errors."""
        error = Exception("Network connection failed")
        code = self.error_handler._determine_error_code(error)
        assert code == "network_error"
    
    def test_determine_error_code_timeout(self):
        """Test error code determination for timeout errors."""
        error = Exception("Request timeout occurred")
        code = self.error_handler._determine_error_code(error)
        assert code == "timeout_error"
    
    def test_determine_error_code_service_unavailable(self):
        """Test error code determination for service unavailable errors."""
        error = Exception("Service unavailable")
        code = self.error_handler._determine_error_code(error)
        assert code == "service_unavailable"
    
    def test_determine_error_code_model_not_found(self):
        """Test error code determination for model not found errors."""
        error = Exception("Model not found")
        code = self.error_handler._determine_error_code(error)
        assert code == "model_not_found"
    
    def test_determine_error_code_empty_response(self):
        """Test error code determination for empty response errors."""
        error = Exception("Empty response received")
        code = self.error_handler._determine_error_code(error)
        assert code == "empty_response"
    
    def test_determine_error_code_invalid_category(self):
        """Test error code determination for invalid category errors."""
        error = Exception("Invalid category specified")
        code = self.error_handler._determine_error_code(error)
        assert code == "invalid_category"
    
    def test_determine_error_code_feedback_storage(self):
        """Test error code determination for feedback storage errors."""
        error = Exception("Feedback storage failed")
        code = self.error_handler._determine_error_code(error)
        assert code == "feedback_storage_error"
    
    def test_determine_error_code_general(self):
        """Test error code determination for general errors."""
        error = Exception("Some unexpected error")
        code = self.error_handler._determine_error_code(error)
        assert code == "general_error"
    
    @patch('sys.exit')
    @patch('sys.stderr', new_callable=StringIO)
    def test_handle_error_with_exit(self, mock_stderr, mock_exit):
        """Test handling error with exit."""
        error = Exception("Test error")
        
        self.error_handler.handle_error(
            error,
            error_code="general_error",
            exit_on_error=True,
            test_param="Test error"
        )
        
        mock_exit.assert_called_once_with(EXIT_GENERAL_ERROR)
        output = mock_stderr.getvalue()
        assert "❌ Error:" in output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_handle_error_without_exit(self, mock_stderr):
        """Test handling error without exit."""
        error = Exception("Test error")
        
        exit_code = self.error_handler.handle_error(
            error,
            error_code="general_error",
            exit_on_error=False,
            test_param="Test error"
        )
        
        assert exit_code == EXIT_GENERAL_ERROR
        output = mock_stderr.getvalue()
        assert "❌ Error:" in output
    
    @patch('sys.exit')
    @patch('sys.stderr', new_callable=StringIO)
    def test_validate_and_handle_error_success(self, mock_stderr, mock_exit):
        """Test validation with successful condition."""
        exit_code = self.error_handler.validate_and_handle_error(
            condition=True,
            error_code="invalid_category",
            exit_on_error=True
        )
        
        assert exit_code is None
        mock_exit.assert_not_called()
        assert mock_stderr.getvalue() == ""
    
    @patch('sys.exit')
    @patch('sys.stderr', new_callable=StringIO)
    def test_validate_and_handle_error_failure(self, mock_stderr, mock_exit):
        """Test validation with failed condition."""
        self.error_handler.validate_and_handle_error(
            condition=False,
            error_code="invalid_category",
            exit_on_error=True,
            category="invalid",
            available_categories="general, programming"
        )
        
        mock_exit.assert_called_once()
        output = mock_stderr.getvalue()
        assert "❌ Error:" in output
        assert "Invalid joke category" in output


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    @patch('joke_cli.error_handler.ErrorHandler')
    def test_get_error_handler(self, mock_error_handler_class):
        """Test getting the global error handler."""
        mock_handler = Mock()
        mock_error_handler_class.return_value = mock_handler
        
        # Clear any existing global handler
        import joke_cli.error_handler
        joke_cli.error_handler._global_error_handler = None
        
        handler1 = get_error_handler(debug=True)
        handler2 = get_error_handler(debug=False)
        
        # Should return the same instance (singleton behavior)
        assert handler1 == handler2
        mock_error_handler_class.assert_called_once_with(debug=True)
    
    @patch('joke_cli.error_handler.get_error_handler')
    def test_handle_error_function(self, mock_get_handler):
        """Test the global handle_error function."""
        mock_handler = Mock()
        mock_handler.handle_error.return_value = 42
        mock_get_handler.return_value = mock_handler
        
        error = Exception("Test error")
        result = handle_error(
            error,
            error_code="test_error",
            exit_on_error=False,
            param="value"
        )
        
        assert result == 42
        mock_handler.handle_error.assert_called_once_with(
            error,
            "test_error",
            False,
            param="value"
        )
    
    @patch('sys.exit')
    @patch('joke_cli.error_handler.get_error_handler')
    def test_display_error_message_function(self, mock_get_handler, mock_exit):
        """Test the global display_error_message function."""
        mock_handler = Mock()
        mock_handler.format_error_message.return_value = {
            "message": "Test message",
            "guidance": ["Step 1"],
            "exit_code": 42
        }
        mock_get_handler.return_value = mock_handler
        
        display_error_message(
            "test_error",
            exit_on_error=True,
            param="value"
        )
        
        mock_handler.format_error_message.assert_called_once_with("test_error", param="value")
        mock_handler.display_error.assert_called_once_with("Test message", ["Step 1"])
        mock_exit.assert_called_once_with(42)
    
    @patch('joke_cli.error_handler.get_error_handler')
    def test_validate_condition_function(self, mock_get_handler):
        """Test the global validate_condition function."""
        mock_handler = Mock()
        mock_handler.validate_and_handle_error.return_value = 42
        mock_get_handler.return_value = mock_handler
        
        result = validate_condition(
            condition=False,
            error_code="test_error",
            exit_on_error=False,
            param="value"
        )
        
        assert result == 42
        mock_handler.validate_and_handle_error.assert_called_once_with(
            False,
            "test_error",
            False,
            param="value"
        )


class TestLoggingSetup:
    """Test logging configuration."""
    
    def test_logging_setup_with_debug(self):
        """Test logging setup with debug enabled."""
        error_handler = ErrorHandler(logger_name="test_debug", debug=True)
        
        assert error_handler.logger.level == logging.DEBUG
        assert error_handler.debug is True
    
    def test_logging_setup_without_debug(self):
        """Test logging setup without debug."""
        error_handler = ErrorHandler(logger_name="test_no_debug", debug=False)
        
        # Logger level might be DEBUG due to existing handlers, but debug flag should be False
        assert error_handler.debug is False
    
    @patch('logging.FileHandler', side_effect=Exception("Cannot create file"))
    @patch('pathlib.Path.mkdir')
    def test_logging_setup_file_creation_failure(self, mock_mkdir, mock_file_handler):
        """Test logging setup when file creation fails."""
        # Should not raise exception, just continue without file logging
        error_handler = ErrorHandler(logger_name="test", debug=True)
        
        assert error_handler.logger.level == logging.DEBUG
        assert error_handler.debug is True


class TestErrorMessageFormatting:
    """Test error message formatting edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_format_with_all_parameters(self):
        """Test formatting with all required parameters."""
        result = self.error_handler.format_error_message(
            "access_denied",
            model_id="test-model"
        )
        
        assert "test-model" in result["message"]
        assert "bedrock" in result["guidance"][1].lower()
        assert result["exit_code"] > 0
    
    def test_format_with_partial_parameters(self):
        """Test formatting with some missing parameters."""
        result = self.error_handler.format_error_message(
            "model_not_found",
            model_id="test-model"
            # Missing default_model parameter
        )
        
        assert "test-model" in result["message"]
        assert len(result["guidance"]) > 0
        # Should handle missing parameters gracefully
    
    def test_format_guidance_with_parameters(self):
        """Test that guidance messages are formatted with parameters."""
        result = self.error_handler.format_error_message(
            "timeout_error",
            timeout=30
        )
        
        assert "30 seconds" in result["message"]
        assert len(result["guidance"]) > 0