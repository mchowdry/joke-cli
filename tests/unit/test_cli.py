"""
Unit tests for the CLI module.

Tests argument parsing, validation, and main application flow.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import argparse
import sys
from io import StringIO

from joke_cli.cli import (
    create_argument_parser,
    parse_arguments,
    validate_arguments,
    display_statistics,
    main
)
from joke_cli.models import JokeResponse
from joke_cli.config import AVAILABLE_CATEGORIES


class TestArgumentParser:
    """Test cases for argument parser creation and configuration."""
    
    def test_create_argument_parser_basic(self):
        """Test basic argument parser creation."""
        parser = create_argument_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "joke"
        assert "Generate jokes using AWS Bedrock AI models" in parser.description
    
    def test_parser_has_all_expected_arguments(self):
        """Test that parser includes all required arguments."""
        parser = create_argument_parser()
        
        # Get all argument names
        arg_names = []
        for action in parser._actions:
            if action.option_strings:
                arg_names.extend(action.option_strings)
        
        expected_args = [
            "--category", "-c",
            "--profile", "-p", 
            "--no-feedback",
            "--stats", "-s",
            "--version",
            "--help", "-h"  # Added by argparse automatically
        ]
        
        for expected_arg in expected_args:
            assert expected_arg in arg_names
    
    def test_category_argument_choices(self):
        """Test that category argument has correct choices."""
        parser = create_argument_parser()
        
        # Find the category action
        category_action = None
        for action in parser._actions:
            if "--category" in action.option_strings:
                category_action = action
                break
        
        assert category_action is not None
        assert category_action.choices == AVAILABLE_CATEGORIES


class TestArgumentParsing:
    """Test cases for argument parsing functionality."""
    
    def test_parse_no_arguments(self):
        """Test parsing with no arguments."""
        args = parse_arguments([])
        
        assert args.category is None
        assert args.profile is None
        assert args.no_feedback is False
        assert args.stats is False
    
    def test_parse_category_argument(self):
        """Test parsing category argument."""
        args = parse_arguments(["--category", "programming"])
        
        assert args.category == "programming"
        assert args.profile is None
        assert args.no_feedback is False
        assert args.stats is False
    
    def test_parse_category_short_form(self):
        """Test parsing category argument with short form."""
        args = parse_arguments(["-c", "dad-jokes"])
        
        assert args.category == "dad-jokes"
    
    def test_parse_profile_argument(self):
        """Test parsing profile argument."""
        args = parse_arguments(["--profile", "my-profile"])
        
        assert args.profile == "my-profile"
        assert args.category is None
    
    def test_parse_profile_short_form(self):
        """Test parsing profile argument with short form."""
        args = parse_arguments(["-p", "test-profile"])
        
        assert args.profile == "test-profile"
    
    def test_parse_no_feedback_flag(self):
        """Test parsing no-feedback flag."""
        args = parse_arguments(["--no-feedback"])
        
        assert args.no_feedback is True
        assert args.category is None
        assert args.profile is None
    
    def test_parse_stats_flag(self):
        """Test parsing stats flag."""
        args = parse_arguments(["--stats"])
        
        assert args.stats is True
        assert args.category is None
    
    def test_parse_stats_short_form(self):
        """Test parsing stats flag with short form."""
        args = parse_arguments(["-s"])
        
        assert args.stats is True
    
    def test_parse_multiple_arguments(self):
        """Test parsing multiple arguments together."""
        args = parse_arguments([
            "--category", "programming",
            "--profile", "my-profile",
            "--no-feedback"
        ])
        
        assert args.category == "programming"
        assert args.profile == "my-profile"
        assert args.no_feedback is True
        assert args.stats is False
    
    def test_parse_invalid_category(self):
        """Test parsing with invalid category."""
        with pytest.raises(SystemExit):
            parse_arguments(["--category", "invalid-category"])
    
    def test_parse_help_flag(self):
        """Test that help flag causes system exit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(["--help"])
        
        # Help should exit with code 0
        assert exc_info.value.code == 0
    
    def test_parse_version_flag(self):
        """Test that version flag causes system exit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(["--version"])
        
        # Version should exit with code 0
        assert exc_info.value.code == 0


class TestArgumentValidation:
    """Test cases for argument validation."""
    
    def test_validate_valid_arguments(self):
        """Test validation with valid arguments."""
        args = argparse.Namespace(
            category="programming",
            profile="my-profile",
            no_feedback=False,
            stats=False
        )
        
        # Should not raise any exception
        validate_arguments(args)
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_validate_stats_with_category_fails(self, mock_stderr):
        """Test that stats with category fails validation."""
        args = argparse.Namespace(
            category="programming",
            profile=None,
            no_feedback=False,
            stats=True
        )
        
        with pytest.raises(SystemExit) as exc_info:
            validate_arguments(args)
        
        assert exc_info.value.code == 2  # EXIT_INVALID_ARGUMENTS
        error_output = mock_stderr.getvalue()
        assert "Cannot specify --category with --stats option" in error_output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_validate_stats_with_no_feedback_fails(self, mock_stderr):
        """Test that stats with no-feedback fails validation."""
        args = argparse.Namespace(
            category=None,
            profile=None,
            no_feedback=True,
            stats=True
        )
        
        with pytest.raises(SystemExit) as exc_info:
            validate_arguments(args)
        
        assert exc_info.value.code == 2  # EXIT_INVALID_ARGUMENTS
        error_output = mock_stderr.getvalue()
        assert "Cannot specify --no-feedback with --stats option" in error_output
    
    def test_validate_stats_only_succeeds(self):
        """Test that stats-only arguments pass validation."""
        args = argparse.Namespace(
            category=None,
            profile=None,
            no_feedback=False,
            stats=True
        )
        
        # Should not raise any exception
        validate_arguments(args)
    
    def test_validate_no_feedback_with_category_succeeds(self):
        """Test that no-feedback with category passes validation."""
        args = argparse.Namespace(
            category="programming",
            profile=None,
            no_feedback=True,
            stats=False
        )
        
        # Should not raise any exception
        validate_arguments(args)
    
    @patch('boto3.Session')
    def test_validate_valid_aws_profile(self, mock_session_class):
        """Test validation with valid AWS profile."""
        mock_session = Mock()
        mock_session.get_credentials.return_value = Mock()
        mock_session_class.return_value = mock_session
        
        args = argparse.Namespace(
            category=None,
            profile="valid-profile",
            no_feedback=False,
            stats=False
        )
        
        # Should not raise any exception
        validate_arguments(args)
        mock_session_class.assert_called_once_with(profile_name="valid-profile")
    
    @patch('boto3.Session')
    @patch('sys.stderr', new_callable=StringIO)
    def test_validate_invalid_aws_profile(self, mock_stderr, mock_session_class):
        """Test validation with invalid AWS profile."""
        mock_session = Mock()
        mock_session.get_credentials.side_effect = Exception("Profile not found")
        mock_session_class.return_value = mock_session
        
        args = argparse.Namespace(
            category=None,
            profile="invalid-profile",
            no_feedback=False,
            stats=False
        )
        
        with pytest.raises(SystemExit) as exc_info:
            validate_arguments(args)
        
        assert exc_info.value.code == 3  # EXIT_AWS_CREDENTIALS_ERROR
        error_output = mock_stderr.getvalue()
        assert "AWS profile 'invalid-profile' not found" in error_output


class TestDisplayStatistics:
    """Test cases for statistics display functionality."""
    
    @patch('joke_cli.cli.JokeService')
    def test_display_statistics_success(self, mock_joke_service_class):
        """Test successful statistics display."""
        # Setup mock
        mock_service = Mock()
        mock_service.format_statistics_output.return_value = "Test statistics output"
        mock_joke_service_class.return_value = mock_service
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            display_statistics()
        
        # Verify output
        output = mock_stdout.getvalue()
        assert "Test statistics output" in output
        mock_service.format_statistics_output.assert_called_once()
    
    @patch('joke_cli.cli.JokeService')
    @patch('sys.stderr', new_callable=StringIO)
    def test_display_statistics_error(self, mock_stderr, mock_joke_service_class):
        """Test statistics display with error."""
        # Setup mock to raise exception
        mock_service = Mock()
        mock_service.format_statistics_output.side_effect = Exception("Test error")
        mock_joke_service_class.return_value = mock_service
        
        # Test that it exits with error
        with pytest.raises(SystemExit) as exc_info:
            display_statistics()
        
        # Verify error handling uses enhanced error handler
        assert exc_info.value.code == 1
        error_output = mock_stderr.getvalue()
        assert "❌ Error:" in error_output


class TestMainFunction:
    """Test cases for the main application function."""
    
    @patch('joke_cli.cli.JokeService')
    def test_main_basic_joke_generation(self, mock_joke_service_class):
        """Test basic joke generation flow."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "programming")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke output"
        mock_service.prompt_for_feedback.return_value = (4, "Great joke!")
        mock_service.collect_user_feedback.return_value = True
        mock_joke_service_class.return_value = mock_service
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(["--category", "programming"])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        
        # Verify calls
        mock_service.generate_joke.assert_called_once_with(
            category="programming",
            aws_profile=None
        )
        mock_service.format_joke_output.assert_called_once_with(mock_response)
        mock_service.prompt_for_feedback.assert_called_once()
        mock_service.collect_user_feedback.assert_called_once_with(
            mock_response, 4, "Great joke!"
        )
        
        # Verify output
        output = mock_stdout.getvalue()
        assert "Formatted joke output" in output
        assert "Thanks for your feedback!" in output
    
    @patch('joke_cli.cli.JokeService')
    def test_main_no_feedback_flag(self, mock_joke_service_class):
        """Test main function with no-feedback flag."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_joke_service_class.return_value = mock_service
        
        # Run with no-feedback flag
        with patch('sys.stdout', new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main(["--no-feedback"])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        
        # Verify feedback methods were not called
        mock_service.prompt_for_feedback.assert_not_called()
        mock_service.collect_user_feedback.assert_not_called()
    
    @patch('joke_cli.cli.display_statistics')
    def test_main_stats_flag(self, mock_display_stats):
        """Test main function with stats flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--stats"])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        mock_display_stats.assert_called_once()
    
    @patch('joke_cli.cli.JokeService')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_joke_generation_error(self, mock_stderr, mock_joke_service_class):
        """Test main function with joke generation error."""
        # Setup mock to return error response
        mock_service = Mock()
        mock_response = JokeResponse.create_error("Test error", "programming")
        mock_service.generate_joke.return_value = mock_response
        mock_joke_service_class.return_value = mock_service
        
        # Test that it exits with success (error was handled gracefully)
        with pytest.raises(SystemExit) as exc_info:
            main(["--category", "programming"])
        
        # Should exit with success code (error was handled gracefully)
        assert exc_info.value.code == 0
        error_output = mock_stderr.getvalue()
        assert "❌ Error: Test error" in error_output
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_invalid_arguments(self, mock_stderr):
        """Test main function with invalid arguments."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--stats", "--category", "programming"])
        
        # Should exit with invalid arguments error code
        assert exc_info.value.code == 2
        error_output = mock_stderr.getvalue()
        assert "Cannot specify --category with --stats option" in error_output
    
    @patch('joke_cli.cli.JokeService')
    def test_main_keyboard_interrupt(self, mock_joke_service_class):
        """Test main function handling keyboard interrupt."""
        # Setup mock to raise KeyboardInterrupt
        mock_service = Mock()
        mock_service.generate_joke.side_effect = KeyboardInterrupt()
        mock_joke_service_class.return_value = mock_service
        
        # Test that it handles KeyboardInterrupt gracefully
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with pytest.raises(SystemExit) as exc_info:
                main([])
        
        # Should exit with code 130 (standard for SIGINT)
        assert exc_info.value.code == 130
        error_output = mock_stderr.getvalue()
        assert "Operation cancelled by user." in error_output
    
    @patch('joke_cli.cli.JokeService')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_unexpected_error(self, mock_stderr, mock_joke_service_class):
        """Test main function handling unexpected errors."""
        # Setup mock to raise unexpected exception
        mock_service = Mock()
        mock_service.generate_joke.side_effect = RuntimeError("Unexpected error")
        mock_joke_service_class.return_value = mock_service
        
        # Test that it handles unexpected errors gracefully
        with pytest.raises(SystemExit) as exc_info:
            main([])
        
        # Should exit with general error code
        assert exc_info.value.code == 1
        error_output = mock_stderr.getvalue()
        assert "❌ Error:" in error_output
        assert "unexpected error occurred" in error_output.lower()
    
    @patch('joke_cli.cli.JokeService')
    def test_main_feedback_save_failure(self, mock_joke_service_class):
        """Test main function when feedback save fails."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_service.prompt_for_feedback.return_value = (3, None)
        mock_service.collect_user_feedback.return_value = False  # Simulate save failure
        mock_joke_service_class.return_value = mock_service
        
        # Capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main([])
        
        # Should still exit with success code
        assert exc_info.value.code == 0
        
        # Verify warning message
        error_output = mock_stderr.getvalue()
        assert "⚠️  Warning:" in error_output
        assert "Could not save feedback" in error_output
    
    @patch('joke_cli.cli.JokeService')
    def test_main_feedback_skipped(self, mock_joke_service_class):
        """Test main function when user skips feedback."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_service.prompt_for_feedback.return_value = (None, None)  # User skipped
        mock_joke_service_class.return_value = mock_service
        
        # Run main function
        with patch('sys.stdout', new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main([])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        
        # Verify collect_user_feedback was not called
        mock_service.collect_user_feedback.assert_not_called()
    
    @patch('joke_cli.cli.JokeService')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_feedback_keyboard_interrupt(self, mock_stderr, mock_joke_service_class):
        """Test main function when user interrupts during feedback."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_service.prompt_for_feedback.side_effect = KeyboardInterrupt()
        mock_joke_service_class.return_value = mock_service
        
        # Run main function
        with patch('sys.stdout', new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main([])
        
        # Should exit with success code (feedback interruption is not an error)
        assert exc_info.value.code == 0
        
        # Verify feedback skipped message
        error_output = mock_stderr.getvalue()
        assert "Feedback skipped" in error_output
    
    @patch('os.environ.get')
    @patch('joke_cli.cli.JokeService')
    def test_main_debug_mode_enabled(self, mock_joke_service_class, mock_env_get):
        """Test main function with debug mode enabled."""
        # Setup environment variable
        mock_env_get.return_value = "true"
        
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_service.prompt_for_feedback.return_value = (None, None)
        mock_joke_service_class.return_value = mock_service
        
        # Run main function
        with patch('sys.stdout', new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main([])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        
        # Verify environment variable was checked
        mock_env_get.assert_called_with("JOKE_CLI_DEBUG", "")
    
    @patch('joke_cli.cli.JokeService')
    def test_main_complete_feedback_workflow(self, mock_joke_service_class):
        """Test complete feedback collection workflow in main function."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Why do programmers prefer dark mode?", "programming")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "🎭 Joke of the Day 🎭\n\nWhy do programmers prefer dark mode?\n\nCategory: Programming"
        mock_service.prompt_for_feedback.return_value = (5, "Excellent joke!")
        mock_service.collect_user_feedback.return_value = True
        mock_joke_service_class.return_value = mock_service
        
        # Capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(["--category", "programming"])
        
        # Should exit with success code
        assert exc_info.value.code == 0
        
        # Verify complete workflow
        mock_service.generate_joke.assert_called_once_with(
            category="programming",
            aws_profile=None
        )
        mock_service.format_joke_output.assert_called_once_with(mock_response)
        mock_service.prompt_for_feedback.assert_called_once()
        mock_service.collect_user_feedback.assert_called_once_with(
            mock_response, 5, "Excellent joke!"
        )
        
        # Verify output includes joke and feedback confirmation
        output = mock_stdout.getvalue()
        assert "🎭 Joke of the Day 🎭" in output
        assert "Why do programmers prefer dark mode?" in output
        assert "Thanks for your feedback!" in output
    
    @patch('joke_cli.cli.JokeService')
    def test_main_feedback_workflow_with_different_ratings(self, mock_joke_service_class):
        """Test feedback workflow with different rating values."""
        test_cases = [
            (1, "Not funny"),
            (2, "Meh"),
            (3, None),  # No comment
            (4, "Pretty good"),
            (5, "Hilarious!")
        ]
        
        for rating, comment in test_cases:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("Test joke", "general")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "Formatted joke"
            mock_service.prompt_for_feedback.return_value = (rating, comment)
            mock_service.collect_user_feedback.return_value = True
            mock_joke_service_class.return_value = mock_service
            
            # Run main function
            with patch('sys.stdout', new_callable=StringIO):
                with pytest.raises(SystemExit) as exc_info:
                    main([])
            
            # Should exit with success code
            assert exc_info.value.code == 0
            
            # Verify feedback was collected with correct values
            mock_service.collect_user_feedback.assert_called_once_with(
                mock_response, rating, comment
            )
    
    @patch('joke_cli.cli.JokeService')
    def test_main_feedback_error_handling(self, mock_joke_service_class):
        """Test main function handles feedback errors gracefully."""
        # Setup mocks
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_service.prompt_for_feedback.side_effect = Exception("Feedback error")
        mock_joke_service_class.return_value = mock_service
        
        # Capture output
        with patch('sys.stdout', new_callable=StringIO):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main([])
        
        # Should still exit with success code (feedback errors don't fail the app)
        assert exc_info.value.code == 0
        
        # Verify warning message
        error_output = mock_stderr.getvalue()
        assert "⚠️  Warning:" in error_output
        assert "Error collecting feedback" in error_output


class TestArgumentParsingEdgeCases:
    """Test edge cases and error conditions in argument parsing."""
    
    def test_empty_string_arguments(self):
        """Test parsing with empty string arguments."""
        # Empty strings should be treated as valid values where applicable
        args = parse_arguments(["--profile", ""])
        assert args.profile == ""
    
    def test_multiple_category_arguments_last_wins(self):
        """Test that multiple category arguments use the last one."""
        args = parse_arguments(["--category", "programming", "--category", "puns"])
        assert args.category == "puns"
    
    def test_case_sensitive_category_validation(self):
        """Test that category validation is case sensitive."""
        with pytest.raises(SystemExit):
            parse_arguments(["--category", "Programming"])  # Capital P should fail


if __name__ == "__main__":
    pytest.main([__file__])