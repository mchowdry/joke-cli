"""
Integration tests for feedback collection workflow.

Tests the complete feedback collection process from joke generation
through user interaction to storage and statistics.
"""

import pytest
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory
from pathlib import Path

from joke_cli.joke_service import JokeService
from joke_cli.feedback_storage import FeedbackStorage
from joke_cli.models import JokeResponse
from joke_cli.cli import main


@pytest.mark.integration
class TestFeedbackIntegration:
    """Integration tests for the complete feedback workflow."""
    
    def test_complete_feedback_workflow_with_real_storage(self):
        """Test complete feedback workflow with real storage backend."""
        with TemporaryDirectory() as temp_dir:
            # Setup real feedback storage in temp directory
            storage_dir = Path(temp_dir)
            feedback_storage = FeedbackStorage(storage_dir)
            
            # Mock Bedrock client to avoid AWS calls
            mock_bedrock_client = Mock()
            mock_bedrock_client.invoke_model.return_value = "Why do programmers prefer dark mode? Because light attracts bugs!"
            
            # Create service with real storage
            service = JokeService(
                bedrock_client=mock_bedrock_client,
                feedback_storage=feedback_storage
            )
            
            # Generate a joke
            joke_response = service.generate_joke(category="programming")
            assert joke_response.success is True
            assert "programmers" in joke_response.joke_text.lower()
            
            # Simulate user feedback collection
            rating = 4
            comment = "Pretty good programming joke!"
            
            # Collect feedback
            success = service.collect_user_feedback(joke_response, rating, comment)
            assert success is True
            
            # Verify feedback was stored
            stats = service.get_feedback_statistics()
            assert stats["total_jokes"] == 1
            assert stats["average_rating"] == 4.0
            assert "programming" in stats["category_stats"]
            assert stats["category_stats"]["programming"]["count"] == 1
            assert stats["category_stats"]["programming"]["avg_rating"] == 4.0
            
            # Generate another joke and collect feedback
            joke_response2 = service.generate_joke(category="general")
            assert joke_response2.success is True
            
            success2 = service.collect_user_feedback(joke_response2, 5, "Excellent!")
            assert success2 is True
            
            # Verify updated statistics
            updated_stats = service.get_feedback_statistics()
            assert updated_stats["total_jokes"] == 2
            assert updated_stats["average_rating"] == 4.5  # (4 + 5) / 2
            assert len(updated_stats["category_stats"]) == 2
    
    @patch('builtins.input')
    def test_interactive_feedback_collection(self, mock_input):
        """Test interactive feedback collection with user input simulation."""
        with TemporaryDirectory() as temp_dir:
            # Setup
            storage_dir = Path(temp_dir)
            feedback_storage = FeedbackStorage(storage_dir)
            mock_bedrock_client = Mock()
            mock_bedrock_client.invoke_model.return_value = "Test joke content"
            
            service = JokeService(
                bedrock_client=mock_bedrock_client,
                feedback_storage=feedback_storage
            )
            
            # Simulate user input: rating 3, comment "Not bad"
            mock_input.side_effect = ['3', 'Not bad']
            
            # Generate joke
            joke_response = service.generate_joke()
            assert joke_response.success is True
            
            # Prompt for feedback (simulated user interaction)
            rating, comment = service.prompt_for_feedback()
            assert rating == 3
            assert comment == "Not bad"
            
            # Collect the feedback
            success = service.collect_user_feedback(joke_response, rating, comment)
            assert success is True
            
            # Verify storage
            stats = service.get_feedback_statistics()
            assert stats["total_jokes"] == 1
            assert stats["average_rating"] == 3.0
    
    @patch('builtins.input')
    def test_feedback_skip_workflow(self, mock_input):
        """Test workflow when user skips feedback."""
        with TemporaryDirectory() as temp_dir:
            # Setup
            storage_dir = Path(temp_dir)
            feedback_storage = FeedbackStorage(storage_dir)
            mock_bedrock_client = Mock()
            mock_bedrock_client.invoke_model.return_value = "Skipped joke"
            
            service = JokeService(
                bedrock_client=mock_bedrock_client,
                feedback_storage=feedback_storage
            )
            
            # Simulate user skipping feedback
            mock_input.return_value = 's'
            
            # Generate joke
            joke_response = service.generate_joke()
            assert joke_response.success is True
            
            # User skips feedback
            rating, comment = service.prompt_for_feedback()
            assert rating is None
            assert comment is None
            
            # No feedback should be collected
            # (In real workflow, collect_user_feedback wouldn't be called)
            
            # Verify no feedback in storage
            stats = service.get_feedback_statistics()
            assert stats["total_jokes"] == 0
            assert stats["average_rating"] == 0.0
    
    def test_cli_integration_with_feedback(self):
        """Test CLI integration with feedback collection."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("CLI integration joke", "general")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 CLI integration joke"
            mock_service.prompt_for_feedback.return_value = (4, "Good CLI test")
            mock_service.collect_user_feedback.return_value = True
            mock_service_class.return_value = mock_service
            
            # Run CLI with feedback enabled (default behavior)
            with pytest.raises(SystemExit) as exc_info:
                main([])
            
            # Should exit successfully
            assert exc_info.value.code == 0
            
            # Verify complete workflow was executed
            mock_service.generate_joke.assert_called_once()
            mock_service.format_joke_output.assert_called_once()
            mock_service.prompt_for_feedback.assert_called_once()
            mock_service.collect_user_feedback.assert_called_once_with(
                mock_response, 4, "Good CLI test"
            )
    
    def test_cli_integration_no_feedback(self):
        """Test CLI integration with feedback disabled."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_response = JokeResponse.create_success("No feedback joke", "general")
            mock_service.generate_joke.return_value = mock_response
            mock_service.format_joke_output.return_value = "🎭 No feedback joke"
            mock_service_class.return_value = mock_service
            
            # Run CLI with feedback disabled
            with pytest.raises(SystemExit) as exc_info:
                main(["--no-feedback"])
            
            # Should exit successfully
            assert exc_info.value.code == 0
            
            # Verify joke generation but no feedback collection
            mock_service.generate_joke.assert_called_once()
            mock_service.format_joke_output.assert_called_once()
            mock_service.prompt_for_feedback.assert_not_called()
            mock_service.collect_user_feedback.assert_not_called()
    
    def test_statistics_display_integration(self):
        """Test statistics display integration."""
        with patch('joke_cli.cli.JokeService') as mock_service_class:
            # Setup mocks
            mock_service = Mock()
            mock_stats = {
                "total_jokes": 5,
                "average_rating": 3.8,
                "category_stats": {
                    "programming": {"count": 3, "avg_rating": 4.0},
                    "general": {"count": 2, "avg_rating": 3.5}
                }
            }
            mock_service.format_statistics_output.return_value = "📊 Test Statistics"
            mock_service_class.return_value = mock_service
            
            # Run CLI with stats flag
            with pytest.raises(SystemExit) as exc_info:
                main(["--stats"])
            
            # Should exit successfully
            assert exc_info.value.code == 0
            
            # Verify statistics display
            mock_service.format_statistics_output.assert_called_once()
            
            # Verify no joke generation when showing stats
            mock_service.generate_joke.assert_not_called()
            mock_service.prompt_for_feedback.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])