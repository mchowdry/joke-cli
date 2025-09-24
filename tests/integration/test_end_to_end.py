"""
End-to-end integration tests for the Joke CLI application.

Tests the complete application workflow from CLI entry point through
joke generation, feedback collection, and statistics display.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from tempfile import TemporaryDirectory
from pathlib import Path
from io import StringIO

from joke_cli.cli import main, initialize_application, cleanup_application, orchestrate_joke_generation
from joke_cli.models import JokeResponse
from joke_cli.config import EXIT_SUCCESS, EXIT_INVALID_ARGUMENTS, EXIT_USER_CANCELLED


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for complete application workflows."""
    
    def test_complete_joke_generation_workflow(self):
        """Test complete joke generation workflow from CLI entry to exit."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("End-to-end test joke", "programming")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 Joke of the Day 🎭\n\nEnd-to-end test joke\n\nCategory: Programming"
            mock_service.prompt_for_feedback.return_value = (4, "Great test!")
            mock_service.collect_user_feedback.return_value = True
            mock_service_class.return_value = mock_service
            
            # Capture stdout
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main(["--category", "programming"])
                
                # Verify successful exit
                assert exc_info.value.code == EXIT_SUCCESS
                
                # Verify output contains joke
                output = mock_stdout.getvalue()
                assert "End-to-end test joke" in output
                assert "Thanks for your feedback!" in output
                
                # Verify complete workflow execution
                mock_service.generate_joke.assert_called_once_with(
                    category="programming",
                    aws_profile=None
                )
                mock_service.format_joke_output.assert_called_once_with(mock_response)
                mock_service.prompt_for_feedback.assert_called_once()
                mock_service.collect_user_feedback.assert_called_once_with(
                    mock_response, 4, "Great test!"
                )
    
    def test_no_feedback_workflow(self):
        """Test complete workflow with feedback disabled."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("No feedback joke", "general")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 No feedback joke"
            mock_service_class.return_value = mock_service
            
            # Capture stdout
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main(["--no-feedback"])
                
                # Verify successful exit
                assert exc_info.value.code == EXIT_SUCCESS
                
                # Verify output contains joke but no feedback message
                output = mock_stdout.getvalue()
                assert "No feedback joke" in output
                assert "Thanks for your feedback!" not in output
                
                # Verify no feedback collection
                mock_service.prompt_for_feedback.assert_not_called()
                mock_service.collect_user_feedback.assert_not_called()
    
    def test_statistics_workflow(self):
        """Test statistics display workflow."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_service.format_statistics_output.return_value = "📊 Test Statistics\nTotal jokes: 5"
            mock_service_class.return_value = mock_service
            
            # Capture stdout
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main(["--stats"])
                
                # Verify successful exit
                assert exc_info.value.code == EXIT_SUCCESS
                
                # Verify statistics output
                output = mock_stdout.getvalue()
                assert "📊 Test Statistics" in output
                assert "Total jokes: 5" in output
                
                # Verify no joke generation
                mock_service.generate_joke.assert_not_called()
    
    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup service to raise exception
            mock_service = Mock()
            mock_service.generate_joke.side_effect = Exception("Test error")
            mock_service_class.return_value = mock_service
            
            # Mock error handler
            with patch('joke_cli.cli.get_error_handler') as mock_error_handler:
                mock_handler = Mock()
                mock_error_handler.return_value = mock_handler
                
                with pytest.raises(SystemExit) as exc_info:
                    main([])
                
                # Should exit successfully even with error (error is handled)
                assert exc_info.value.code == EXIT_SUCCESS
                
                # Verify error handler was called
                mock_handler.handle_error.assert_called()
    
    def test_keyboard_interrupt_handling(self):
        """Test graceful handling of keyboard interrupt."""
        with patch('joke_cli.cli.parse_arguments') as mock_parse:
            # Simulate KeyboardInterrupt during argument parsing
            mock_parse.side_effect = KeyboardInterrupt()
            
            # Capture stderr
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main([])
                
                # Verify user cancellation exit code
                assert exc_info.value.code == EXIT_USER_CANCELLED
                
                # Verify user-friendly message
                output = mock_stderr.getvalue()
                assert "Operation cancelled by user" in output
    
    def test_invalid_arguments_handling(self):
        """Test handling of invalid command line arguments."""
        # Test invalid category
        with patch('sys.stderr', new_callable=StringIO):
            with pytest.raises(SystemExit) as exc_info:
                main(["--category", "invalid-category"])
            
            # Should exit with argument error code
            assert exc_info.value.code == 2  # argparse default for invalid arguments
    
    def test_aws_profile_validation(self):
        """Test AWS profile validation in workflow."""
        with patch('boto3.Session') as mock_session_class:
            # Setup mock to simulate invalid profile
            mock_session = Mock()
            mock_session.get_credentials.side_effect = Exception("Profile not found")
            mock_session_class.return_value = mock_session
            
            with patch('joke_cli.cli.display_error_message') as mock_display_error:
                with pytest.raises(SystemExit):
                    main(["--profile", "invalid-profile"])
                
                # Verify error display was called
                mock_display_error.assert_called_once()
    
    def test_application_initialization(self):
        """Test application initialization function."""
        with patch('logging.basicConfig') as mock_logging:
            with patch.dict(os.environ, {'JOKE_CLI_DEBUG': 'true'}):
                initialize_application()
                
                # Verify logging was configured
                mock_logging.assert_called_once()
                call_args = mock_logging.call_args
                assert call_args[1]['level'] == 10  # DEBUG level
    
    def test_application_cleanup(self):
        """Test application cleanup function."""
        with patch('sys.stdout') as mock_stdout, \
             patch('sys.stderr') as mock_stderr:
            
            cleanup_application()
            
            # Verify streams were flushed
            mock_stdout.flush.assert_called_once()
            mock_stderr.flush.assert_called_once()
    
    def test_orchestrate_joke_generation_success(self):
        """Test joke generation orchestration with successful flow."""
        from argparse import Namespace
        
        # Create mock arguments
        args = Namespace(
            category="programming",
            profile=None,
            no_feedback=False
        )
        
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("Orchestration test", "programming")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 Orchestration test"
            mock_service.prompt_for_feedback.return_value = (5, "Excellent!")
            mock_service.collect_user_feedback.return_value = True
            mock_service_class.return_value = mock_service
            
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                exit_code = orchestrate_joke_generation(args)
                
                # Verify successful exit code
                assert exit_code == EXIT_SUCCESS
                
                # Verify output
                output = mock_stdout.getvalue()
                assert "Orchestration test" in output
                assert "Thanks for your feedback!" in output
    
    def test_orchestrate_joke_generation_failure(self):
        """Test joke generation orchestration with failed joke generation."""
        from argparse import Namespace
        
        # Create mock arguments
        args = Namespace(
            category="programming",
            profile=None,
            no_feedback=False
        )
        
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup service to return failed response
            mock_service = Mock()
            mock_response = JokeResponse.create_error("Generation failed", "programming")
            mock_service.generate_joke.return_value = mock_response
            mock_service_class.return_value = mock_service
            
            with patch('joke_cli.cli.get_error_handler') as mock_error_handler:
                mock_handler = Mock()
                mock_error_handler.return_value = mock_handler
                
                exit_code = orchestrate_joke_generation(args)
                
                # Should still exit successfully (error is handled gracefully)
                assert exit_code == EXIT_SUCCESS
                
                # Verify error display was called
                mock_handler.display_error.assert_called_once()
    
    def test_feedback_collection_keyboard_interrupt(self):
        """Test handling of keyboard interrupt during feedback collection."""
        from argparse import Namespace
        
        # Create mock arguments
        args = Namespace(
            category="general",
            profile=None,
            no_feedback=False
        )
        
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("Interrupt test", "general")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 Interrupt test"
            mock_service.prompt_for_feedback.side_effect = KeyboardInterrupt()
            mock_service_class.return_value = mock_service
            
            with patch('sys.stdout', new_callable=StringIO), \
                 patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                
                exit_code = orchestrate_joke_generation(args)
                
                # Should still exit successfully
                assert exit_code == EXIT_SUCCESS
                
                # Verify feedback skipped message
                error_output = mock_stderr.getvalue()
                assert "Feedback skipped" in error_output
    
    def test_real_storage_integration(self):
        """Test integration with real storage backend."""
        with TemporaryDirectory() as temp_dir:
            # Set up environment to use temp directory for storage
            with patch.dict(os.environ, {'HOME': temp_dir}):
                # Create real service with mocked Bedrock client and fresh storage
                from joke_cli.joke_service import JokeService
                from joke_cli.feedback_storage import FeedbackStorage
                from pathlib import Path
                
                # Create fresh storage in temp directory
                storage_dir = Path(temp_dir) / ".joke_cli"
                storage_dir.mkdir(parents=True, exist_ok=True)
                fresh_storage = FeedbackStorage(storage_dir)
                
                mock_bedrock_client = Mock()
                mock_bedrock_client.invoke_model.return_value = "Real storage test joke"
                
                real_service = JokeService(
                    bedrock_client=mock_bedrock_client,
                    feedback_storage=fresh_storage
                )
                
                with patch('joke_cli.cli.JokeService') as mock_service_class:
                    mock_service_class.return_value = real_service
                    
                    # Mock user input for feedback
                    with patch('builtins.input', side_effect=['4', 'Good integration test']):
                        with patch('sys.stdout', new_callable=StringIO):
                            with pytest.raises(SystemExit) as exc_info:
                                main([])
                            
                            # Verify successful execution
                            assert exc_info.value.code == EXIT_SUCCESS
                            
                            # Verify feedback was actually stored
                            stats = real_service.get_feedback_statistics()
                            assert stats["total_jokes"] == 1
                            assert stats["average_rating"] == 4.0


@pytest.mark.integration
class TestCLIArgumentIntegration:
    """Integration tests for CLI argument handling in complete workflows."""
    
    def test_all_argument_combinations(self):
        """Test various argument combinations in complete workflows."""
        test_cases = [
            # (args, should_generate_joke, should_collect_feedback, should_show_stats)
            ([], True, True, False),
            (["--category", "programming"], True, True, False),
            (["--no-feedback"], True, False, False),
            (["--category", "general", "--no-feedback"], True, False, False),
            (["--stats"], False, False, True),
            (["--profile", "test-profile"], True, True, False),
        ]
        
        for args, should_generate, should_collect_feedback, should_show_stats in test_cases:
            with patch('joke_cli.cli.JokeService') as mock_service_class:
                # Setup mocks
                mock_service = Mock()
                mock_response = JokeResponse.create_success("Test joke", "general")
                mock_service.generate_joke.return_value = mock_response
                mock_service.format_joke_output.return_value = "🎭 Test joke"
                mock_service.prompt_for_feedback.return_value = (3, "OK")
                mock_service.collect_user_feedback.return_value = True
                mock_service.format_statistics_output.return_value = "📊 Stats"
                mock_service_class.return_value = mock_service
                
                # Handle profile validation for profile tests
                if "--profile" in args:
                    with patch('boto3.Session') as mock_session_class:
                        mock_session = Mock()
                        mock_session.get_credentials.return_value = Mock()
                        mock_session_class.return_value = mock_session
                        
                        with patch('sys.stdout', new_callable=StringIO):
                            with pytest.raises(SystemExit) as exc_info:
                                main(args)
                else:
                    with patch('sys.stdout', new_callable=StringIO):
                        with pytest.raises(SystemExit) as exc_info:
                            main(args)
                
                # Verify successful exit
                assert exc_info.value.code == EXIT_SUCCESS
                
                # Verify expected behavior
                if should_generate:
                    mock_service.generate_joke.assert_called()
                else:
                    mock_service.generate_joke.assert_not_called()
                
                if should_collect_feedback:
                    mock_service.prompt_for_feedback.assert_called()
                else:
                    mock_service.prompt_for_feedback.assert_not_called()
                
                if should_show_stats:
                    mock_service.format_statistics_output.assert_called()
                else:
                    mock_service.format_statistics_output.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])