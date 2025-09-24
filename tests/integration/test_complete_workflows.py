"""
Integration tests for complete user workflows.

Tests end-to-end user scenarios including joke generation, feedback collection,
statistics viewing, and error recovery workflows.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from tempfile import TemporaryDirectory
from pathlib import Path
from io import StringIO

from joke_cli.cli import main
from joke_cli.joke_service import JokeService
from joke_cli.feedback_storage import FeedbackStorage
from joke_cli.models import JokeResponse, BedrockConfig
from tests.fixtures.mock_data import (
    MockJokes,
    MockBedrockResponses,
    MockAWSResponses,
    MockUserInput,
    MockFeedbackData,
    TestDataBuilder,
    create_successful_joke_workflow_mocks
)


@pytest.mark.integration
class TestCompleteUserWorkflows:
    """Integration tests for complete user workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = None
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir:
            self.temp_dir.cleanup()
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_first_time_user_workflow(self, mock_input, mock_session_class):
        """Test complete workflow for a first-time user."""
        with TemporaryDirectory() as temp_dir:
            # Setup AWS mocks
            mock_session = MockAWSResponses.create_session_mock()
            mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
            
            joke_text = MockJokes.PROGRAMMING_JOKES[0]
            mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
            mock_bedrock_client.invoke_model.return_value = mock_response
            
            mock_session.client.return_value = mock_bedrock_client
            mock_session_class.return_value = mock_session
            
            # Setup user input for feedback
            mock_input.side_effect = ['5', 'Great first joke!']
            
            # Setup environment to use temp directory
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit) as exc_info:
                        main(['--category', 'programming'])
                
                # Verify successful execution
                assert exc_info.value.code == 0
                
                # Verify output contains joke and feedback confirmation
                output = mock_stdout.getvalue()
                assert joke_text in output
                assert "Thanks for your feedback!" in output
                
                # Verify feedback was stored
                storage_dir = Path(temp_dir) / ".joke_cli"
                assert storage_dir.exists()
                
                feedback_file = storage_dir / "feedback.json"
                assert feedback_file.exists()
                
                with open(feedback_file) as f:
                    feedback_data = json.load(f)
                
                assert len(feedback_data["feedback_entries"]) == 1
                assert feedback_data["feedback_entries"][0]["rating"] == 5
                assert feedback_data["feedback_entries"][0]["user_comment"] == "Great first joke!"
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_returning_user_workflow(self, mock_input, mock_session_class):
        """Test workflow for a returning user with existing feedback."""
        with TemporaryDirectory() as temp_dir:
            # Setup existing feedback data
            storage_dir = Path(temp_dir) / ".joke_cli"
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            existing_feedback = {
                "feedback_entries": MockFeedbackData.create_sample_feedback_entries()[:3],
                "stats": {
                    "total_jokes": 3,
                    "average_rating": 4.0
                }
            }
            
            feedback_file = storage_dir / "feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(existing_feedback, f)
            
            # Setup AWS mocks
            mock_session = MockAWSResponses.create_session_mock()
            mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
            
            joke_text = MockJokes.DAD_JOKES[0]
            mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
            mock_bedrock_client.invoke_model.return_value = mock_response
            
            mock_session.client.return_value = mock_bedrock_client
            mock_session_class.return_value = mock_session
            
            # Setup user input for new feedback
            mock_input.side_effect = ['3', 'Not bad']
            
            # Setup environment to use temp directory
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit) as exc_info:
                        main(['--category', 'dad-jokes'])
                
                # Verify successful execution
                assert exc_info.value.code == 0
                
                # Verify new feedback was added
                with open(feedback_file) as f:
                    updated_feedback = json.load(f)
                
                assert len(updated_feedback["feedback_entries"]) == 4
                assert updated_feedback["feedback_entries"][-1]["rating"] == 3
                assert updated_feedback["feedback_entries"][-1]["user_comment"] == "Not bad"
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_statistics_viewing_workflow(self, mock_session_class):
        """Test workflow for viewing statistics."""
        with TemporaryDirectory() as temp_dir:
            # Setup existing feedback data
            storage_dir = Path(temp_dir) / ".joke_cli"
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            feedback_data = {
                "feedback_entries": MockFeedbackData.create_sample_feedback_entries(),
                "stats": MockFeedbackData.create_feedback_statistics()
            }
            
            feedback_file = storage_dir / "feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f)
            
            # Setup environment to use temp directory
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit) as exc_info:
                        main(['--stats'])
                
                # Verify successful execution
                assert exc_info.value.code == 0
                
                # Verify statistics output
                output = mock_stdout.getvalue()
                assert "📊 Feedback Statistics" in output
                assert "Total jokes: 5" in output
                assert "Average rating: 3.6" in output
                assert "programming" in output
                assert "general" in output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_no_feedback_workflow(self, mock_session_class):
        """Test workflow with feedback disabled."""
        # Setup AWS mocks
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
        
        joke_text = MockJokes.PUNS[0]
        mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(['--category', 'puns', '--no-feedback'])
        
        # Verify successful execution
        assert exc_info.value.code == 0
        
        # Verify output contains joke but no feedback prompts
        output = mock_stdout.getvalue()
        assert joke_text in output
        assert "Thanks for your feedback!" not in output
        assert "Rate this joke" not in output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_feedback_skip_workflow(self, mock_input, mock_session_class):
        """Test workflow when user skips feedback."""
        # Setup AWS mocks
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
        
        joke_text = MockJokes.CLEAN_JOKES[0]
        mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # User skips feedback
        mock_input.return_value = 's'
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(['--category', 'clean'])
        
        # Verify successful execution
        assert exc_info.value.code == 0
        
        # Verify output contains joke but no feedback confirmation
        output = mock_stdout.getvalue()
        assert joke_text in output
        assert "Thanks for your feedback!" not in output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_feedback_error_recovery_workflow(self, mock_input, mock_session_class):
        """Test workflow with feedback collection errors and recovery."""
        with TemporaryDirectory() as temp_dir:
            # Setup AWS mocks
            mock_session = MockAWSResponses.create_session_mock()
            mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
            
            joke_text = MockJokes.GENERAL_JOKES[0]
            mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
            mock_bedrock_client.invoke_model.return_value = mock_response
            
            mock_session.client.return_value = mock_bedrock_client
            mock_session_class.return_value = mock_session
            
            # Simulate invalid input followed by valid input
            mock_input.side_effect = ['invalid', '6', '4', 'Good joke after correction']
            
            # Setup environment to use temp directory
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                        with pytest.raises(SystemExit) as exc_info:
                            main([])
                
                # Verify successful execution despite input errors
                assert exc_info.value.code == 0
                
                # Verify joke was displayed
                output = mock_stdout.getvalue()
                assert joke_text in output
                
                # Verify error messages for invalid input
                error_output = mock_stderr.getvalue()
                assert "Invalid rating" in error_output or "Please enter" in error_output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_aws_error_recovery_workflow(self, mock_session_class):
        """Test workflow with AWS errors and graceful handling."""
        # Setup AWS mocks with error
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = Mock()
        
        # Setup credentials test to pass
        mock_bedrock_client.list_foundation_models.return_value = {}
        
        # Setup model invocation to fail
        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}}
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            error_response, 'InvokeModel'
        )
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main(['--category', 'programming'])
        
        # Should exit successfully (error handled gracefully)
        assert exc_info.value.code == 0
        
        # Verify error message was displayed
        error_output = mock_stderr.getvalue()
        assert "❌ Error:" in error_output
        assert "Access denied" in error_output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_keyboard_interrupt_workflow(self, mock_input, mock_session_class):
        """Test workflow with keyboard interrupt handling."""
        # Setup AWS mocks
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
        
        joke_text = MockJokes.PROGRAMMING_JOKES[0]
        mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # User interrupts during feedback
        mock_input.side_effect = KeyboardInterrupt()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main([])
        
        # Should exit successfully (interruption handled gracefully)
        assert exc_info.value.code == 0
        
        # Verify joke was displayed
        output = mock_stdout.getvalue()
        assert joke_text in output
        
        # Verify feedback skipped message
        error_output = mock_stderr.getvalue()
        assert "Feedback skipped" in error_output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_multiple_categories_workflow(self, mock_input, mock_session_class):
        """Test workflow with multiple joke categories."""
        categories_to_test = ['programming', 'dad-jokes', 'puns', 'clean', 'general']
        
        for category in categories_to_test:
            # Setup AWS mocks
            mock_session = MockAWSResponses.create_session_mock()
            mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
            
            joke_text = MockJokes.get_joke_by_category(category)
            mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
            mock_bedrock_client.invoke_model.return_value = mock_response
            
            mock_session.client.return_value = mock_bedrock_client
            mock_session_class.return_value = mock_session
            
            # Setup user input
            mock_input.side_effect = ['4', f'Good {category} joke']
            
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit) as exc_info:
                    main(['--category', category])
            
            # Verify successful execution
            assert exc_info.value.code == 0
            
            # Verify category-specific joke was displayed
            output = mock_stdout.getvalue()
            assert joke_text in output
            assert f"Category: {category.title()}" in output or category in output.lower()
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_aws_profile_workflow(self, mock_session_class):
        """Test workflow with AWS profile specification."""
        # Setup AWS mocks with profile
        mock_session = MockAWSResponses.create_session_mock(profile="production")
        mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
        
        joke_text = MockJokes.GENERAL_JOKES[0]
        mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                main(['--profile', 'production', '--no-feedback'])
        
        # Verify successful execution
        assert exc_info.value.code == 0
        
        # Verify profile was used
        mock_session_class.assert_called_with(profile_name="production")
        
        # Verify joke was displayed
        output = mock_stdout.getvalue()
        assert joke_text in output
    
    def test_invalid_arguments_workflow(self):
        """Test workflow with invalid command line arguments."""
        invalid_argument_cases = [
            (['--category', 'invalid-category'], 2),
            (['--stats', '--category', 'programming'], 2),
            (['--stats', '--no-feedback'], 2),
        ]
        
        for args, expected_exit_code in invalid_argument_cases:
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    main(args)
            
            # Verify appropriate exit code
            assert exc_info.value.code == expected_exit_code
            
            # Verify error message was displayed
            error_output = mock_stderr.getvalue()
            assert len(error_output) > 0  # Some error message should be present


@pytest.mark.integration
class TestDataPersistenceWorkflows:
    """Integration tests for data persistence across sessions."""
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_feedback_persistence_across_sessions(self, mock_input, mock_session_class):
        """Test that feedback persists across multiple application sessions."""
        with TemporaryDirectory() as temp_dir:
            # Setup AWS mocks
            mock_session = MockAWSResponses.create_session_mock()
            mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
            mock_session.client.return_value = mock_bedrock_client
            mock_session_class.return_value = mock_session
            
            # Session 1: Generate joke and provide feedback
            joke_text_1 = MockJokes.PROGRAMMING_JOKES[0]
            mock_response_1 = MockBedrockResponses.create_titan_success_response(joke_text_1)
            mock_bedrock_client.invoke_model.return_value = mock_response_1
            mock_input.side_effect = ['5', 'Excellent!']
            
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with pytest.raises(SystemExit):
                    main(['--category', 'programming'])
            
            # Session 2: Generate another joke and provide feedback
            joke_text_2 = MockJokes.DAD_JOKES[0]
            mock_response_2 = MockBedrockResponses.create_titan_success_response(joke_text_2)
            mock_bedrock_client.invoke_model.return_value = mock_response_2
            mock_input.side_effect = ['3', 'Okay']
            
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with pytest.raises(SystemExit):
                    main(['--category', 'dad-jokes'])
            
            # Session 3: View statistics
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit):
                        main(['--stats'])
            
            # Verify statistics include both sessions
            output = mock_stdout.getvalue()
            assert "Total jokes: 2" in output
            assert "Average rating: 4.0" in output  # (5+3)/2
            assert "programming" in output
            assert "dad-jokes" in output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_statistics_with_no_feedback_data(self, mock_session_class):
        """Test statistics display when no feedback data exists."""
        with TemporaryDirectory() as temp_dir:
            # Setup environment with empty feedback directory
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit) as exc_info:
                        main(['--stats'])
            
            # Should exit successfully
            assert exc_info.value.code == 0
            
            # Verify appropriate message for no data
            output = mock_stdout.getvalue()
            assert ("No feedback data available" in output or 
                    "Total jokes: 0" in output)
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_feedback_storage_error_handling(self, mock_input, mock_session_class):
        """Test handling of feedback storage errors."""
        # Setup AWS mocks
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = MockAWSResponses.create_bedrock_client_mock()
        
        joke_text = MockJokes.GENERAL_JOKES[0]
        mock_response = MockBedrockResponses.create_titan_success_response(joke_text)
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        # Setup user input
        mock_input.side_effect = ['4', 'Good joke']
        
        # Mock feedback storage to fail
        with patch('joke_cli.joke_service.FeedbackStorage') as mock_storage_class:
            mock_storage = Mock()
            mock_storage.save_feedback.return_value = False  # Simulate save failure
            mock_storage_class.return_value = mock_storage
            
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                    with pytest.raises(SystemExit) as exc_info:
                        main([])
            
            # Should still exit successfully
            assert exc_info.value.code == 0
            
            # Verify joke was displayed
            output = mock_stdout.getvalue()
            assert joke_text in output
            
            # Verify warning about feedback save failure
            error_output = mock_stderr.getvalue()
            assert "⚠️  Warning:" in error_output
            assert "Could not save feedback" in error_output


@pytest.mark.integration
class TestPerformanceWorkflows:
    """Integration tests for performance-related workflows."""
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    @patch('builtins.input')
    def test_large_feedback_dataset_workflow(self, mock_input, mock_session_class):
        """Test workflow with large feedback dataset."""
        with TemporaryDirectory() as temp_dir:
            # Setup large feedback dataset
            storage_dir = Path(temp_dir) / ".joke_cli"
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 100 feedback entries
            large_feedback_entries = []
            for i in range(100):
                entry = {
                    "joke_id": f"test-joke-{i}",
                    "joke_text": f"Test joke {i}",
                    "category": ["programming", "general", "dad-jokes"][i % 3],
                    "rating": (i % 5) + 1,
                    "timestamp": f"2024-01-{(i % 30) + 1:02d}T10:00:00Z",
                    "user_comment": f"Comment {i}" if i % 2 == 0 else None
                }
                large_feedback_entries.append(entry)
            
            feedback_data = {
                "feedback_entries": large_feedback_entries,
                "stats": {
                    "total_jokes": 100,
                    "average_rating": 3.0
                }
            }
            
            feedback_file = storage_dir / "feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f)
            
            # Test statistics display with large dataset
            with patch.dict('os.environ', {'HOME': temp_dir}):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    with pytest.raises(SystemExit) as exc_info:
                        main(['--stats'])
            
            # Should handle large dataset successfully
            assert exc_info.value.code == 0
            
            # Verify statistics output
            output = mock_stdout.getvalue()
            assert "Total jokes: 100" in output
    
    @patch('joke_cli.bedrock_client.boto3.Session')
    def test_timeout_handling_workflow(self, mock_session_class):
        """Test workflow with API timeout scenarios."""
        # Setup AWS mocks with timeout
        mock_session = MockAWSResponses.create_session_mock()
        mock_bedrock_client = Mock()
        
        # Setup credentials test to pass
        mock_bedrock_client.list_foundation_models.return_value = {}
        
        # Setup model invocation to timeout
        from botocore.exceptions import ReadTimeoutError
        mock_bedrock_client.invoke_model.side_effect = ReadTimeoutError(
            endpoint_url="test"
        )
        
        mock_session.client.return_value = mock_bedrock_client
        mock_session_class.return_value = mock_session
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with pytest.raises(SystemExit) as exc_info:
                main([])
        
        # Should handle timeout gracefully
        assert exc_info.value.code == 0
        
        # Verify timeout error message
        error_output = mock_stderr.getvalue()
        assert "❌ Error:" in error_output
        assert ("timed out" in error_output.lower() or 
                "network error" in error_output.lower())


if __name__ == "__main__":
    pytest.main([__file__])