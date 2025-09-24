"""
Unit tests for the joke service module.

Tests the core business logic for joke generation, category management,
and feedback collection functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from uuid import uuid4

from joke_cli.joke_service import (
    JokeService, 
    JokeServiceError,
    generate_joke,
    collect_feedback,
    get_feedback_stats,
    format_joke_for_display,
    format_stats_for_display
)
from joke_cli.models import JokeRequest, JokeResponse, FeedbackEntry, BedrockConfig
from joke_cli.bedrock_client import BedrockClientError
from joke_cli.config import AVAILABLE_CATEGORIES


class TestJokeService:
    """Test cases for the JokeService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_bedrock_client = Mock()
        self.mock_feedback_storage = Mock()
        self.service = JokeService(
            bedrock_client=self.mock_bedrock_client,
            feedback_storage=self.mock_feedback_storage
        )
    
    def test_init_with_dependencies(self):
        """Test service initialization with provided dependencies."""
        assert self.service._bedrock_client == self.mock_bedrock_client
        assert self.service._feedback_storage == self.mock_feedback_storage
    
    def test_init_with_default_dependencies(self):
        """Test service initialization with default dependencies."""
        service = JokeService()
        assert service._bedrock_client is None
        assert service._feedback_storage is not None
    
    def test_generate_joke_success(self):
        """Test successful joke generation."""
        # Setup
        category = "programming"
        joke_text = "Why do programmers prefer dark mode? Because light attracts bugs!"
        self.mock_bedrock_client.invoke_model.return_value = joke_text
        
        # Execute
        result = self.service.generate_joke(category=category)
        
        # Verify
        assert result.success is True
        assert result.joke_text == joke_text
        assert result.category == category
        assert result.error_message is None
        assert isinstance(result.joke_id, str)
        assert isinstance(result.timestamp, datetime)
        
        # Verify Bedrock client was called correctly
        self.mock_bedrock_client.invoke_model.assert_called_once()
        call_args = self.mock_bedrock_client.invoke_model.call_args
        assert "programming" in call_args[0][0].lower()  # Prompt should contain category
        assert isinstance(call_args[0][1], BedrockConfig)
    
    def test_generate_joke_random_category(self):
        """Test joke generation with random category selection."""
        # Setup
        joke_text = "A random joke!"
        self.mock_bedrock_client.invoke_model.return_value = joke_text
        
        # Execute
        result = self.service.generate_joke()
        
        # Verify
        assert result.success is True
        assert result.category in AVAILABLE_CATEGORIES
        assert result.joke_text == joke_text
    
    def test_generate_joke_invalid_category(self):
        """Test joke generation with invalid category."""
        # Execute
        result = self.service.generate_joke(category="invalid-category")
        
        # Verify
        assert result.success is False
        assert "Invalid joke category" in result.error_message
        assert "invalid-category" in result.error_message
        assert result.category == "invalid-category"
        
        # Verify Bedrock client was not called
        self.mock_bedrock_client.invoke_model.assert_not_called()
    
    def test_generate_joke_bedrock_error(self):
        """Test joke generation with Bedrock client error."""
        # Setup
        error_message = "Access denied to Bedrock model"
        self.mock_bedrock_client.invoke_model.side_effect = BedrockClientError(error_message)
        
        # Execute
        result = self.service.generate_joke(category="general")
        
        # Verify
        assert result.success is False
        assert result.error_message == error_message
        assert result.category == "general"
    
    def test_generate_joke_unexpected_error(self):
        """Test joke generation with unexpected error."""
        # Setup
        self.mock_bedrock_client.invoke_model.side_effect = Exception("Unexpected error")
        
        # Execute
        result = self.service.generate_joke(category="general")
        
        # Verify
        assert result.success is False
        assert "Unexpected error" in result.error_message
        assert result.category == "general"
    
    def test_generate_joke_empty_response(self):
        """Test joke generation with empty response from model."""
        # Setup
        self.mock_bedrock_client.invoke_model.return_value = ""
        
        # Execute
        result = self.service.generate_joke(category="general")
        
        # Verify
        assert result.success is False
        assert "Generated joke was empty" in result.error_message
    
    def test_clean_joke_text_removes_prefixes(self):
        """Test that joke text cleaning removes common prefixes."""
        test_cases = [
            ("Here's a joke for you: Why did the chicken cross the road?", 
             "Why did the chicken cross the road?"),
            ("Joke: What do you call a fake noodle?", 
             "What do you call a fake noodle?"),
            ("Sure, here's a joke: How do you organize a space party?", 
             "How do you organize a space party?"),
        ]
        
        for input_text, expected in test_cases:
            result = self.service._clean_joke_text(input_text)
            assert result == expected
    
    def test_clean_joke_text_removes_suffixes(self):
        """Test that joke text cleaning removes common suffixes."""
        test_cases = [
            ("Why did the programmer quit? Hope you enjoyed it!", 
             "Why did the programmer quit?"),
            ("What's a computer's favorite snack? Hope that made you smile!", 
             "What's a computer's favorite snack?"),
        ]
        
        for input_text, expected in test_cases:
            result = self.service._clean_joke_text(input_text)
            assert result == expected
    
    def test_clean_joke_text_normalizes_whitespace(self):
        """Test that joke text cleaning normalizes whitespace."""
        input_text = "  Why did the   programmer quit?  \n\n  Because he didn't get arrays!  \n  "
        expected = "Why did the   programmer quit?\nBecause he didn't get arrays!"
        
        result = self.service._clean_joke_text(input_text)
        assert result == expected
    
    def test_format_joke_output_success(self):
        """Test formatting successful joke response for display."""
        joke_response = JokeResponse.create_success(
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "programming"
        )
        
        result = self.service.format_joke_output(joke_response)
        
        assert "🎭 Joke of the Day 🎭" in result
        assert joke_response.joke_text in result
        assert "Category: Programming" in result
    
    def test_format_joke_output_error(self):
        """Test formatting error joke response for display."""
        joke_response = JokeResponse.create_error("Network error occurred")
        
        result = self.service.format_joke_output(joke_response)
        
        assert result == "Error: Network error occurred"
    
    def test_collect_user_feedback_success(self):
        """Test successful feedback collection."""
        # Setup
        joke_response = JokeResponse.create_success("A great joke!", "general")
        rating = 4
        comment = "Pretty funny!"
        
        # Execute
        result = self.service.collect_user_feedback(joke_response, rating, comment)
        
        # Verify
        assert result is True
        self.mock_feedback_storage.save_feedback.assert_called_once()
        
        # Verify the feedback entry
        call_args = self.mock_feedback_storage.save_feedback.call_args[0][0]
        assert call_args.joke_id == joke_response.joke_id
        assert call_args.joke_text == joke_response.joke_text
        assert call_args.category == joke_response.category
        assert call_args.rating == rating
        assert call_args.user_comment == comment
    
    def test_collect_user_feedback_failed_joke(self):
        """Test feedback collection for failed joke response."""
        # Setup
        joke_response = JokeResponse.create_error("Some error")
        
        # Execute
        result = self.service.collect_user_feedback(joke_response, 3)
        
        # Verify
        assert result is False
        self.mock_feedback_storage.save_feedback.assert_not_called()
    
    def test_collect_user_feedback_storage_error(self):
        """Test feedback collection with storage error."""
        # Setup
        joke_response = JokeResponse.create_success("A joke", "general")
        self.mock_feedback_storage.save_feedback.side_effect = Exception("Storage error")
        
        # Execute
        result = self.service.collect_user_feedback(joke_response, 3)
        
        # Verify
        assert result is False
    
    @patch('builtins.input')
    def test_prompt_for_feedback_valid_rating(self, mock_input):
        """Test prompting for feedback with valid rating."""
        # Setup
        mock_input.side_effect = ['4', 'Great joke!']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 4
        assert comment == "Great joke!"
    
    @patch('builtins.input')
    def test_prompt_for_feedback_skip(self, mock_input):
        """Test prompting for feedback with skip option."""
        # Setup
        mock_input.return_value = 's'
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating is None
        assert comment is None
    
    @patch('builtins.input')
    def test_prompt_for_feedback_invalid_then_valid(self, mock_input):
        """Test prompting for feedback with invalid input then valid."""
        # Setup
        mock_input.side_effect = ['invalid', '6', '3', 'Nice!']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 3
        assert comment == "Nice!"
    
    @patch('builtins.input')
    def test_prompt_for_feedback_empty_comment(self, mock_input):
        """Test prompting for feedback with empty comment."""
        # Setup
        mock_input.side_effect = ['5', '']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 5
        assert comment is None
    
    @patch('builtins.input')
    def test_prompt_for_feedback_keyboard_interrupt(self, mock_input):
        """Test prompting for feedback with keyboard interrupt."""
        # Setup
        mock_input.side_effect = KeyboardInterrupt()
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating is None
        assert comment is None
    
    @patch('builtins.input')
    def test_prompt_for_feedback_eof_error(self, mock_input):
        """Test prompting for feedback with EOF error."""
        # Setup
        mock_input.side_effect = EOFError()
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating is None
        assert comment is None
    
    @patch('builtins.input')
    def test_prompt_for_feedback_multiple_invalid_inputs(self, mock_input):
        """Test prompting for feedback with multiple invalid inputs before valid."""
        # Setup - multiple invalid inputs then valid
        mock_input.side_effect = ['abc', '0', '6', 'ten', '3', 'Finally valid!']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 3
        assert comment == "Finally valid!"
    
    @patch('builtins.input')
    def test_prompt_for_feedback_skip_variations(self, mock_input):
        """Test prompting for feedback with different skip variations."""
        test_cases = ['s', 'S', 'skip', 'SKIP', 'Skip']
        
        for skip_input in test_cases:
            mock_input.return_value = skip_input
            
            # Execute
            rating, comment = self.service.prompt_for_feedback()
            
            # Verify
            assert rating is None, f"Failed for skip input: {skip_input}"
            assert comment is None, f"Failed for skip input: {skip_input}"
    
    @patch('builtins.input')
    def test_prompt_for_feedback_whitespace_handling(self, mock_input):
        """Test prompting for feedback with whitespace in inputs."""
        # Setup - inputs with various whitespace
        mock_input.side_effect = ['  4  ', '  Great joke with spaces!  ']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 4
        assert comment == "Great joke with spaces!"
    
    @patch('builtins.input')
    def test_prompt_for_feedback_comment_with_newlines(self, mock_input):
        """Test prompting for feedback with comment containing special characters."""
        # Setup
        mock_input.side_effect = ['5', 'Great joke!\nReally funny!']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify
        assert rating == 5
        assert comment == "Great joke!\nReally funny!"
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_prompt_for_feedback_error_messages_displayed(self, mock_print, mock_input):
        """Test that error messages are displayed for invalid inputs."""
        # Setup - invalid input then valid
        mock_input.side_effect = ['invalid', '3', '']
        
        # Execute
        rating, comment = self.service.prompt_for_feedback()
        
        # Verify rating and comment
        assert rating == 3
        assert comment is None
        
        # Verify error message was printed
        print_calls = [call.args[0] for call in mock_print.call_args_list if call.args]
        error_messages = [msg for msg in print_calls if "❌" in msg]
        assert len(error_messages) > 0, "Expected error message to be displayed"
    
    def test_get_available_categories(self):
        """Test getting available categories."""
        result = self.service.get_available_categories()
        
        assert result == AVAILABLE_CATEGORIES
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_get_feedback_statistics_success(self):
        """Test getting feedback statistics successfully."""
        # Setup
        expected_stats = {
            "total_jokes": 10,
            "average_rating": 3.5,
            "category_stats": {
                "programming": {"count": 5, "avg_rating": 4.0},
                "general": {"count": 5, "avg_rating": 3.0}
            }
        }
        self.mock_feedback_storage.get_feedback_stats.return_value = expected_stats
        
        # Execute
        result = self.service.get_feedback_statistics()
        
        # Verify
        assert result == expected_stats
        self.mock_feedback_storage.get_feedback_stats.assert_called_once()
    
    def test_get_feedback_statistics_error(self):
        """Test getting feedback statistics with storage error."""
        # Setup
        self.mock_feedback_storage.get_feedback_stats.side_effect = Exception("Storage error")
        
        # Execute
        result = self.service.get_feedback_statistics()
        
        # Verify
        assert result == {
            "total_jokes": 0,
            "average_rating": 0.0,
            "category_stats": {}
        }
    
    def test_format_statistics_output_with_data(self):
        """Test formatting statistics output with data."""
        stats = {
            "total_jokes": 15,
            "average_rating": 3.8,
            "category_stats": {
                "programming": {"count": 8, "avg_rating": 4.2},
                "general": {"count": 7, "avg_rating": 3.3}
            }
        }
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 1, 2: 2, 3: 3, 4: 5, 5: 4
        })
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Feedback Statistics" in result
        assert "📈 Total jokes rated: 15" in result
        assert "⭐ Average rating: 3.8/5.0" in result
        assert "Programming: 8 jokes" in result
        assert "General: 7 jokes" in result
        assert "📊 Rating Distribution:" in result
        assert "🏆 Most popular: Programming" in result
        assert "📉 Least popular: General" in result
        assert "🌟 Highest rated: Programming" in result
        assert "💭 Lowest rated: General" in result
    
    def test_format_statistics_output_no_data(self):
        """Test formatting statistics output with no data."""
        stats = {
            "total_jokes": 0,
            "average_rating": 0.0,
            "category_stats": {}
        }
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Feedback Statistics" in result
        assert "No feedback data available yet" in result
    
    def test_format_statistics_output_fetch_stats(self):
        """Test formatting statistics output that fetches stats."""
        # Setup
        expected_stats = {"total_jokes": 5, "average_rating": 4.0, "category_stats": {}}
        self.mock_feedback_storage.get_feedback_stats.return_value = expected_stats
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 0, 2: 0, 3: 1, 4: 2, 5: 2
        })
        
        # Execute (without providing stats)
        result = self.service.format_statistics_output()
        
        # Verify
        assert "📈 Total jokes rated: 5" in result
        self.mock_feedback_storage.get_feedback_stats.assert_called_once()
    
    def test_calculate_rating_distribution(self):
        """Test calculating rating distribution."""
        # Setup mock feedback entries with proper UUIDs
        feedback_entries = [
            FeedbackEntry.create("550e8400-e29b-41d4-a716-446655440001", "joke1", "general", 5),
            FeedbackEntry.create("550e8400-e29b-41d4-a716-446655440002", "joke2", "programming", 4),
            FeedbackEntry.create("550e8400-e29b-41d4-a716-446655440003", "joke3", "general", 5),
            FeedbackEntry.create("550e8400-e29b-41d4-a716-446655440004", "joke4", "programming", 3),
            FeedbackEntry.create("550e8400-e29b-41d4-a716-446655440005", "joke5", "general", 4),
        ]
        self.mock_feedback_storage.get_all_feedback.return_value = feedback_entries
        
        # Execute
        result = self.service._calculate_rating_distribution()
        
        # Verify
        expected = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        assert result == expected
    
    def test_calculate_rating_distribution_empty(self):
        """Test calculating rating distribution with no data."""
        self.mock_feedback_storage.get_all_feedback.return_value = []
        
        result = self.service._calculate_rating_distribution()
        
        expected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        assert result == expected
    
    def test_calculate_rating_distribution_error(self):
        """Test calculating rating distribution with storage error."""
        self.mock_feedback_storage.get_all_feedback.side_effect = Exception("Storage error")
        
        result = self.service._calculate_rating_distribution()
        
        assert result == {}
    
    def test_format_statistics_output_single_category(self):
        """Test formatting statistics output with single category."""
        stats = {
            "total_jokes": 5,
            "average_rating": 4.2,
            "category_stats": {
                "programming": {"count": 5, "avg_rating": 4.2}
            }
        }
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 0, 2: 0, 3: 1, 4: 2, 5: 2
        })
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Feedback Statistics" in result
        assert "Programming: 5 jokes (100.0%)" in result
        # Should not show most/least popular for single category
        assert "🏆 Most popular:" not in result
        assert "📉 Least popular:" not in result
    
    def test_format_statistics_output_multiple_categories_same_rating(self):
        """Test formatting statistics output with multiple categories having same rating."""
        stats = {
            "total_jokes": 10,
            "average_rating": 4.0,
            "category_stats": {
                "programming": {"count": 5, "avg_rating": 4.0},
                "general": {"count": 5, "avg_rating": 4.0}
            }
        }
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 0, 2: 0, 3: 2, 4: 4, 5: 4
        })
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Feedback Statistics" in result
        assert "Programming: 5 jokes (50.0%)" in result
        assert "General: 5 jokes (50.0%)" in result
        assert "🏆 Most popular:" in result
        assert "📉 Least popular:" in result
        # Should not show best/worst rated when ratings are the same
        assert "🌟 Highest rated:" not in result
        assert "💭 Lowest rated:" not in result
    
    def test_format_statistics_output_rating_distribution_display(self):
        """Test that rating distribution is properly displayed."""
        stats = {
            "total_jokes": 10,
            "average_rating": 3.5,
            "category_stats": {}
        }
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 1, 2: 1, 3: 2, 4: 3, 5: 3
        })
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Rating Distribution:" in result
        assert "⭐⭐⭐⭐⭐ (5):  3 jokes" in result
        assert "⭐⭐⭐⭐ (4):  3 jokes" in result
        assert "⭐⭐⭐ (3):  2 jokes" in result
        assert "⭐⭐ (2):  1 jokes" in result
        assert "⭐ (1):  1 jokes" in result
        assert "30.0%" in result  # 3/10 * 100 for 5-star ratings
    
    def test_format_statistics_output_no_rating_distribution(self):
        """Test formatting statistics output when rating distribution fails."""
        stats = {
            "total_jokes": 5,
            "average_rating": 4.0,
            "category_stats": {}
        }
        
        # Mock rating distribution to return empty dict (error case)
        self.service._calculate_rating_distribution = Mock(return_value={})
        
        result = self.service.format_statistics_output(stats)
        
        assert "📊 Feedback Statistics" in result
        assert "📈 Total jokes rated: 5" in result
        # Should not show rating distribution section when empty
        assert "📊 Rating Distribution:" not in result
    
    def test_calculate_rating_distribution_invalid_ratings(self):
        """Test calculating rating distribution with invalid ratings."""
        # Setup mock feedback entries with some invalid ratings
        feedback_entries = [
            Mock(rating=5),  # Valid
            Mock(rating=0),  # Invalid (too low)
            Mock(rating=6),  # Invalid (too high)
            Mock(rating=3),  # Valid
            Mock(rating=-1), # Invalid (negative)
        ]
        self.mock_feedback_storage.get_all_feedback.return_value = feedback_entries
        
        # Execute
        result = self.service._calculate_rating_distribution()
        
        # Verify - only valid ratings should be counted
        expected = {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}
        assert result == expected
    
    def test_format_statistics_output_comprehensive_display(self):
        """Test comprehensive statistics display with all features."""
        stats = {
            "total_jokes": 20,
            "average_rating": 3.7,
            "category_stats": {
                "programming": {"count": 8, "avg_rating": 4.5},
                "general": {"count": 6, "avg_rating": 3.2},
                "dad-jokes": {"count": 4, "avg_rating": 2.8},
                "puns": {"count": 2, "avg_rating": 4.0}
            }
        }
        
        # Mock rating distribution
        self.service._calculate_rating_distribution = Mock(return_value={
            1: 2, 2: 3, 3: 5, 4: 6, 5: 4
        })
        
        result = self.service.format_statistics_output(stats)
        
        # Verify all sections are present
        assert "📊 Feedback Statistics" in result
        assert "========================================" in result
        assert "📈 Total jokes rated: 20" in result
        assert "⭐ Average rating: 3.7/5.0" in result
        assert "📊 Rating Distribution:" in result
        assert "📂 By Category:" in result
        
        # Verify most/least popular
        assert "🏆 Most popular: Programming (8 jokes)" in result
        assert "📉 Least popular: Puns (2 jokes)" in result
        
        # Verify best/worst rated
        assert "🌟 Highest rated: Programming (4.5/5.0)" in result
        assert "💭 Lowest rated: Dad-Jokes (2.8/5.0)" in result
        
        # Verify percentages are shown
        assert "(40.0%)" in result  # Programming: 8/20 * 100
        assert "(30.0%)" in result  # General: 6/20 * 100
        assert "(20.0%)" in result  # Dad-jokes: 4/20 * 100
        assert "(10.0%)" in result  # Puns: 2/20 * 100
    
    def test_validate_joke_request_valid(self):
        """Test validating a valid joke request."""
        request = JokeRequest(category="programming", model_id="amazon.titan-text-express-v1")
        
        result = self.service.validate_joke_request(request)
        
        assert result is None
    
    def test_validate_joke_request_invalid(self):
        """Test validating an invalid joke request."""
        # Create a request that bypasses validation during init
        request = JokeRequest.__new__(JokeRequest)
        request.category = "invalid-category"
        request.aws_profile = None
        request.model_id = "amazon.titan-text-express-v1"
        
        result = self.service.validate_joke_request(request)
        
        assert result is not None
        assert "Invalid category" in result


class TestModuleLevelFunctions:
    """Test cases for module-level convenience functions."""
    
    @patch('joke_cli.joke_service.get_default_service')
    def test_generate_joke_function(self, mock_get_service):
        """Test the module-level generate_joke function."""
        # Setup
        mock_service = Mock()
        mock_response = JokeResponse.create_success("Test joke", "general")
        mock_service.generate_joke.return_value = mock_response
        mock_get_service.return_value = mock_service
        
        # Execute
        result = generate_joke(category="general")
        
        # Verify
        assert result == mock_response
        mock_service.generate_joke.assert_called_once_with("general", None, None)
    
    @patch('joke_cli.joke_service.get_default_service')
    def test_collect_feedback_function(self, mock_get_service):
        """Test the module-level collect_feedback function."""
        # Setup
        mock_service = Mock()
        mock_service.collect_user_feedback.return_value = True
        mock_get_service.return_value = mock_service
        
        joke_response = JokeResponse.create_success("Test joke", "general")
        
        # Execute
        result = collect_feedback(joke_response, 4, "Great!")
        
        # Verify
        assert result is True
        mock_service.collect_user_feedback.assert_called_once_with(joke_response, 4, "Great!")
    
    @patch('joke_cli.joke_service.get_default_service')
    def test_get_feedback_stats_function(self, mock_get_service):
        """Test the module-level get_feedback_stats function."""
        # Setup
        mock_service = Mock()
        expected_stats = {"total_jokes": 5, "average_rating": 3.5}
        mock_service.get_feedback_statistics.return_value = expected_stats
        mock_get_service.return_value = mock_service
        
        # Execute
        result = get_feedback_stats()
        
        # Verify
        assert result == expected_stats
        mock_service.get_feedback_statistics.assert_called_once()
    
    @patch('joke_cli.joke_service.get_default_service')
    def test_format_joke_for_display_function(self, mock_get_service):
        """Test the module-level format_joke_for_display function."""
        # Setup
        mock_service = Mock()
        mock_service.format_joke_output.return_value = "Formatted joke"
        mock_get_service.return_value = mock_service
        
        joke_response = JokeResponse.create_success("Test joke", "general")
        
        # Execute
        result = format_joke_for_display(joke_response)
        
        # Verify
        assert result == "Formatted joke"
        mock_service.format_joke_output.assert_called_once_with(joke_response)
    
    @patch('joke_cli.joke_service.get_default_service')
    def test_format_stats_for_display_function(self, mock_get_service):
        """Test the module-level format_stats_for_display function."""
        # Setup
        mock_service = Mock()
        mock_service.format_statistics_output.return_value = "Formatted stats"
        mock_get_service.return_value = mock_service
        
        stats = {"total_jokes": 5}
        
        # Execute
        result = format_stats_for_display(stats)
        
        # Verify
        assert result == "Formatted stats"
        mock_service.format_statistics_output.assert_called_once_with(stats)


@pytest.mark.integration
class TestJokeServiceIntegration:
    """Integration tests for joke service with real dependencies."""
    
    def test_joke_service_with_real_feedback_storage(self):
        """Test joke service with real feedback storage."""
        from joke_cli.feedback_storage import FeedbackStorage
        from tempfile import TemporaryDirectory
        from pathlib import Path
        
        with TemporaryDirectory() as temp_dir:
            # Setup
            storage_dir = Path(temp_dir)
            feedback_storage = FeedbackStorage(storage_dir)
            mock_bedrock_client = Mock()
            mock_bedrock_client.invoke_model.return_value = "A test joke!"
            
            service = JokeService(
                bedrock_client=mock_bedrock_client,
                feedback_storage=feedback_storage
            )
            
            # Generate joke
            joke_response = service.generate_joke(category="general")
            assert joke_response.success is True
            
            # Collect feedback
            success = service.collect_user_feedback(joke_response, 4, "Good joke!")
            assert success is True
            
            # Get statistics
            stats = service.get_feedback_statistics()
            assert stats["total_jokes"] == 1
            assert stats["average_rating"] == 4.0
            
            # Format statistics
            formatted = service.format_statistics_output(stats)
            assert "Total jokes rated: 1" in formatted
            assert "Average rating: 4.0/5.0" in formatted