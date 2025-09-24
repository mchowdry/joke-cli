"""
Comprehensive error handling utilities for the Joke CLI application.

This module provides centralized error handling, logging configuration,
and user-friendly error message formatting with actionable guidance.
"""

import logging
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import ERROR_MESSAGES, EXIT_SUCCESS


class JokeCliError(Exception):
    """Base exception class for Joke CLI application errors."""
    
    def __init__(self, message: str, error_code: str = "general_error", 
                 exit_code: int = 1, guidance: Optional[List[str]] = None):
        """
        Initialize a JokeCliError.
        
        Args:
            message: Human-readable error message
            error_code: Internal error code for categorization
            exit_code: System exit code to use
            guidance: Optional list of guidance strings for the user
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.exit_code = exit_code
        self.guidance = guidance or []


class ErrorHandler:
    """Centralized error handling and logging for the application."""
    
    def __init__(self, logger_name: str = "joke_cli", debug: bool = False):
        """
        Initialize the error handler.
        
        Args:
            logger_name: Name for the logger instance
            debug: Whether to enable debug logging
        """
        self.logger = self._setup_logging(logger_name, debug)
        self.debug = debug
    
    def _setup_logging(self, logger_name: str, debug: bool) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            logger_name: Name for the logger
            debug: Whether to enable debug logging
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(logger_name)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Set log level
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Create console handler for errors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler for debug logs if debug is enabled
        if debug:
            try:
                log_dir = Path.home() / ".joke_cli" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "joke_cli.log"
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            except Exception as e:
                # If we can't create log file, just continue without it
                logger.warning(f"Could not create log file: {e}")
        
        return logger
    
    def format_error_message(self, error_code: str, **kwargs) -> Dict[str, Any]:
        """
        Format an error message with guidance.
        
        Args:
            error_code: Error code to look up in ERROR_MESSAGES
            **kwargs: Format parameters for the error message
            
        Returns:
            Dictionary with formatted message, guidance, and exit code
        """
        if error_code not in ERROR_MESSAGES:
            return {
                "message": f"Unknown error: {error_code}",
                "guidance": ["Please report this issue to the developers."],
                "exit_code": 1
            }
        
        error_info = ERROR_MESSAGES[error_code]
        
        # Format message with provided parameters
        try:
            formatted_message = error_info["message"].format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing format parameter for error {error_code}: {e}")
            formatted_message = error_info["message"]
        
        # Format guidance with provided parameters
        formatted_guidance = []
        for guidance_line in error_info["guidance"]:
            try:
                formatted_guidance.append(guidance_line.format(**kwargs))
            except KeyError:
                formatted_guidance.append(guidance_line)
        
        return {
            "message": formatted_message,
            "guidance": formatted_guidance,
            "exit_code": error_info["exit_code"]
        }
    
    def handle_error(self, error: Exception, error_code: Optional[str] = None, 
                    exit_on_error: bool = True, **format_kwargs) -> Optional[int]:
        """
        Handle an error with appropriate logging and user messaging.
        
        Args:
            error: The exception that occurred
            error_code: Optional error code to use for formatting
            exit_on_error: Whether to exit the application after handling
            **format_kwargs: Additional format parameters for error messages
            
        Returns:
            Exit code if not exiting, None if exiting
        """
        # Log the full error details
        self.logger.error(f"Error occurred: {error}", exc_info=self.debug)
        
        # Determine error code if not provided
        if error_code is None:
            error_code = self._determine_error_code(error)
        
        # Format and display error message
        error_info = self.format_error_message(error_code, **format_kwargs)
        self.display_error(error_info["message"], error_info["guidance"])
        
        exit_code = error_info["exit_code"]
        
        if exit_on_error:
            sys.exit(exit_code)
        
        return exit_code
    
    def _determine_error_code(self, error: Exception) -> str:
        """
        Determine the appropriate error code based on the exception type and message.
        
        Args:
            error: The exception to analyze
            
        Returns:
            Appropriate error code string
        """
        error_str = str(error).lower()
        
        # Check for specific error patterns
        if "credentials" in error_str or "no credentials" in error_str:
            return "no_credentials"
        elif "access denied" in error_str or "unauthorized" in error_str:
            return "access_denied"
        elif "throttling" in error_str or "rate limit" in error_str:
            return "rate_limit"
        elif "network" in error_str or "connection" in error_str:
            return "network_error"
        elif "timeout" in error_str:
            return "timeout_error"
        elif "service unavailable" in error_str:
            return "service_unavailable"
        elif "not found" in error_str and "model" in error_str:
            return "model_not_found"
        elif "empty" in error_str and "response" in error_str:
            return "empty_response"
        elif "invalid" in error_str and "category" in error_str:
            return "invalid_category"
        elif "feedback" in error_str and "storage" in error_str:
            return "feedback_storage_error"
        else:
            return "general_error"
    
    def display_error(self, message: str, guidance: List[str]) -> None:
        """
        Display a formatted error message with guidance to the user.
        
        Args:
            message: Main error message
            guidance: List of guidance strings
        """
        print(f"❌ Error: {message}", file=sys.stderr)
        
        if guidance:
            print("\n💡 How to fix this:", file=sys.stderr)
            for line in guidance:
                print(f"   {line}", file=sys.stderr)
    
    def display_warning(self, message: str) -> None:
        """
        Display a warning message to the user.
        
        Args:
            message: Warning message to display
        """
        print(f"⚠️  Warning: {message}", file=sys.stderr)
        self.logger.warning(message)
    
    def display_info(self, message: str) -> None:
        """
        Display an informational message to the user.
        
        Args:
            message: Info message to display
        """
        print(f"ℹ️  {message}")
        self.logger.info(message)
    
    def validate_and_handle_error(self, condition: bool, error_code: str, 
                                 exit_on_error: bool = True, **format_kwargs) -> Optional[int]:
        """
        Validate a condition and handle error if it fails.
        
        Args:
            condition: Condition to validate (True = valid, False = error)
            error_code: Error code to use if condition fails
            exit_on_error: Whether to exit on error
            **format_kwargs: Format parameters for error message
            
        Returns:
            Exit code if condition fails and not exiting, None otherwise
        """
        if not condition:
            error_info = self.format_error_message(error_code, **format_kwargs)
            self.display_error(error_info["message"], error_info["guidance"])
            
            if exit_on_error:
                sys.exit(error_info["exit_code"])
            
            return error_info["exit_code"]
        
        return None


# Global error handler instance
_global_error_handler = None

def get_error_handler(debug: bool = False) -> ErrorHandler:
    """
    Get the global error handler instance.
    
    Args:
        debug: Whether to enable debug logging
        
    Returns:
        ErrorHandler instance
    """
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(debug=debug)
    return _global_error_handler

def handle_error(error: Exception, error_code: Optional[str] = None, 
                exit_on_error: bool = True, **format_kwargs) -> Optional[int]:
    """
    Convenience function to handle errors using the global error handler.
    
    Args:
        error: The exception that occurred
        error_code: Optional error code to use
        exit_on_error: Whether to exit after handling
        **format_kwargs: Format parameters for error messages
        
    Returns:
        Exit code if not exiting, None if exiting
    """
    handler = get_error_handler()
    return handler.handle_error(error, error_code, exit_on_error, **format_kwargs)

def display_error_message(error_code: str, exit_on_error: bool = True, **format_kwargs) -> Optional[int]:
    """
    Display an error message by error code.
    
    Args:
        error_code: Error code to display
        exit_on_error: Whether to exit after displaying
        **format_kwargs: Format parameters for error message
        
    Returns:
        Exit code if not exiting, None if exiting
    """
    handler = get_error_handler()
    error_info = handler.format_error_message(error_code, **format_kwargs)
    handler.display_error(error_info["message"], error_info["guidance"])
    
    if exit_on_error:
        sys.exit(error_info["exit_code"])
    
    return error_info["exit_code"]

def validate_condition(condition: bool, error_code: str, 
                      exit_on_error: bool = True, **format_kwargs) -> Optional[int]:
    """
    Validate a condition and display error if it fails.
    
    Args:
        condition: Condition to validate
        error_code: Error code to use if validation fails
        exit_on_error: Whether to exit on validation failure
        **format_kwargs: Format parameters for error message
        
    Returns:
        Exit code if validation fails and not exiting, None otherwise
    """
    handler = get_error_handler()
    return handler.validate_and_handle_error(condition, error_code, exit_on_error, **format_kwargs)