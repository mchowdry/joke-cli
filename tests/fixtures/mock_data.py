"""
Mock data and fixtures for Joke CLI application tests.

Provides consistent test data, mock responses, and helper functions
for creating test scenarios across unit and integration tests.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import Mock

from joke_cli.models import JokeResponse, FeedbackEntry, BedrockConfig


class MockJokes:
    """Collection of mock jokes for testing."""
    
    PROGRAMMING_JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
        "Why do Java developers wear glasses? Because they can't C#!",
        "What's a programmer's favorite hangout place? Foo Bar.",
        "Why did the programmer quit his job? He didn't get arrays."
    ]
    
    GENERAL_JOKES = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? He was outstanding in his field!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't eggs tell jokes? They'd crack each other up!"
    ]
    
    DAD_JOKES = [
        "I'm reading a book about anti-gravity. It's impossible to put down!",
        "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!",
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "What do you call a factory that makes okay products? A satisfactory!"
    ]
    
    PUNS = [
        "I wondered why the baseball kept getting bigger. Then it hit me.",
        "A bicycle can't stand on its own because it's two-tired.",
        "What do you call a dinosaur that crashes his car? Tyrannosaurus Wrecks!",
        "I used to hate facial hair, but then it grew on me.",
        "The graveyard is so crowded, people are dying to get in!"
    ]
    
    CLEAN_JOKES = [
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus.",
        "Why did the cookie go to the doctor? Because it felt crumbly!",
        "What do you call a sleeping bull? A bulldozer!",
        "Why don't some couples go to the gym? Because some relationships don't work out!",
        "What did one wall say to the other wall? I'll meet you at the corner!"
    ]
    
    @classmethod
    def get_joke_by_category(cls, category: str) -> str:
        """Get a random joke from the specified category."""
        category_map = {
            "programming": cls.PROGRAMMING_JOKES,
            "general": cls.GENERAL_JOKES,
            "dad-jokes": cls.DAD_JOKES,
            "puns": cls.PUNS,
            "clean": cls.CLEAN_JOKES
        }
        
        jokes = category_map.get(category, cls.GENERAL_JOKES)
        return jokes[0]  # Return first joke for consistency in tests
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all available joke categories."""
        return ["programming", "general", "dad-jokes", "puns", "clean"]


class MockBedrockResponses:
    """Mock AWS Bedrock API responses for testing."""
    
    @staticmethod
    def create_titan_success_response(joke_text: str) -> Dict[str, Any]:
        """Create a successful Titan model response."""
        return {
            'body': Mock(**{
                'read.return_value': json.dumps({
                    'results': [{'outputText': joke_text}]
                }).encode()
            })
        }
    
    @staticmethod
    def create_claude_success_response(joke_text: str) -> Dict[str, Any]:
        """Create a successful Claude model response."""
        return {
            'body': Mock(**{
                'read.return_value': json.dumps({
                    'completion': joke_text
                }).encode()
            })
        }
    
    @staticmethod
    def create_empty_response() -> Dict[str, Any]:
        """Create an empty response for testing error handling."""
        return {
            'body': Mock(**{
                'read.return_value': json.dumps({
                    'results': [{'outputText': ''}]
                }).encode()
            })
        }
    
    @staticmethod
    def create_invalid_json_response() -> Dict[str, Any]:
        """Create an invalid JSON response for testing error handling."""
        return {
            'body': Mock(**{
                'read.return_value': b"invalid json content"
            })
        }
    
    @staticmethod
    def create_missing_field_response() -> Dict[str, Any]:
        """Create a response missing required fields."""
        return {
            'body': Mock(**{
                'read.return_value': json.dumps({
                    'results': []  # Missing outputText
                }).encode()
            })
        }


class MockFeedbackData:
    """Mock feedback data for testing."""
    
    @staticmethod
    def create_sample_feedback_entries() -> List[Dict[str, Any]]:
        """Create sample feedback entries for testing."""
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        return [
            {
                "joke_id": "test-joke-1",
                "joke_text": MockJokes.PROGRAMMING_JOKES[0],
                "category": "programming",
                "rating": 5,
                "timestamp": base_time.isoformat(),
                "user_comment": "Excellent programming joke!"
            },
            {
                "joke_id": "test-joke-2", 
                "joke_text": MockJokes.GENERAL_JOKES[0],
                "category": "general",
                "rating": 3,
                "timestamp": (base_time.replace(hour=11)).isoformat(),
                "user_comment": None
            },
            {
                "joke_id": "test-joke-3",
                "joke_text": MockJokes.DAD_JOKES[0],
                "category": "dad-jokes", 
                "rating": 4,
                "timestamp": (base_time.replace(hour=12)).isoformat(),
                "user_comment": "Classic dad joke!"
            },
            {
                "joke_id": "test-joke-4",
                "joke_text": MockJokes.PUNS[0],
                "category": "puns",
                "rating": 2,
                "timestamp": (base_time.replace(hour=13)).isoformat(),
                "user_comment": "Not my style"
            },
            {
                "joke_id": "test-joke-5",
                "joke_text": MockJokes.CLEAN_JOKES[0],
                "category": "clean",
                "rating": 4,
                "timestamp": (base_time.replace(hour=14)).isoformat(),
                "user_comment": "Nice clean humor"
            }
        ]
    
    @staticmethod
    def create_feedback_statistics() -> Dict[str, Any]:
        """Create sample feedback statistics for testing."""
        return {
            "total_jokes": 5,
            "average_rating": 3.6,  # (5+3+4+2+4)/5
            "category_stats": {
                "programming": {"count": 1, "avg_rating": 5.0},
                "general": {"count": 1, "avg_rating": 3.0},
                "dad-jokes": {"count": 1, "avg_rating": 4.0},
                "puns": {"count": 1, "avg_rating": 2.0},
                "clean": {"count": 1, "avg_rating": 4.0}
            },
            "rating_distribution": {
                "1": 0, "2": 1, "3": 1, "4": 2, "5": 1
            },
            "recent_feedback": [
                {
                    "joke_text": MockJokes.CLEAN_JOKES[0][:50] + "...",
                    "rating": 4,
                    "timestamp": "2024-01-15T14:00:00Z"
                }
            ]
        }


class MockAWSResponses:
    """Mock AWS service responses for testing."""
    
    @staticmethod
    def create_credentials_mock() -> Mock:
        """Create a mock AWS credentials object."""
        mock_credentials = Mock()
        mock_credentials.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_credentials.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_credentials.token = None
        return mock_credentials
    
    @staticmethod
    def create_session_mock(profile: Optional[str] = None) -> Mock:
        """Create a mock boto3 session."""
        mock_session = Mock()
        mock_session.profile_name = profile
        mock_session.get_credentials.return_value = MockAWSResponses.create_credentials_mock()
        return mock_session
    
    @staticmethod
    def create_bedrock_client_mock() -> Mock:
        """Create a mock Bedrock client."""
        mock_client = Mock()
        mock_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {
                    'modelId': 'amazon.titan-text-express-v1',
                    'modelName': 'Titan Text Express',
                    'providerName': 'Amazon'
                },
                {
                    'modelId': 'anthropic.claude-v2',
                    'modelName': 'Claude v2',
                    'providerName': 'Anthropic'
                }
            ]
        }
        return mock_client


class MockErrorResponses:
    """Mock error responses for testing error handling."""
    
    @staticmethod
    def create_client_error(error_code: str, message: str) -> Dict[str, Any]:
        """Create a mock ClientError response."""
        return {
            'Error': {
                'Code': error_code,
                'Message': message
            }
        }
    
    @staticmethod
    def get_common_aws_errors() -> Dict[str, Dict[str, Any]]:
        """Get common AWS error responses for testing."""
        return {
            'access_denied': MockErrorResponses.create_client_error(
                'AccessDeniedException', 
                'User is not authorized to perform this action'
            ),
            'throttling': MockErrorResponses.create_client_error(
                'ThrottlingException',
                'Rate exceeded'
            ),
            'service_unavailable': MockErrorResponses.create_client_error(
                'ServiceUnavailableException',
                'Service is temporarily unavailable'
            ),
            'validation_error': MockErrorResponses.create_client_error(
                'ValidationException',
                'Invalid input parameters'
            ),
            'resource_not_found': MockErrorResponses.create_client_error(
                'ResourceNotFoundException',
                'The requested resource was not found'
            ),
            'invalid_profile': MockErrorResponses.create_client_error(
                'InvalidUserID.NotFound',
                'The AWS profile was not found'
            )
        }


class TestDataBuilder:
    """Builder class for creating test data objects."""
    
    @staticmethod
    def create_joke_response(
        success: bool = True,
        joke_text: str = None,
        category: str = "general",
        error_message: str = None
    ) -> JokeResponse:
        """Create a JokeResponse for testing."""
        if success:
            return JokeResponse.create_success(
                joke_text or MockJokes.get_joke_by_category(category),
                category
            )
        else:
            return JokeResponse.create_error(
                error_message or "Test error",
                category
            )
    
    @staticmethod
    def create_feedback_entry(
        joke_text: str = None,
        category: str = "general",
        rating: int = 3,
        comment: str = None
    ) -> FeedbackEntry:
        """Create a FeedbackEntry for testing."""
        return FeedbackEntry(
            joke_id="test-joke-id",
            joke_text=joke_text or MockJokes.get_joke_by_category(category),
            category=category,
            rating=rating,
            timestamp=datetime.now(timezone.utc),
            user_comment=comment
        )
    
    @staticmethod
    def create_bedrock_config(
        model_id: str = "amazon.titan-text-express-v1",
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> BedrockConfig:
        """Create a BedrockConfig for testing."""
        return BedrockConfig(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )


class MockUserInput:
    """Mock user input scenarios for testing interactive features."""
    
    @staticmethod
    def create_feedback_inputs() -> Dict[str, List[str]]:
        """Create various user input scenarios for feedback collection."""
        return {
            'positive_feedback': ['5', 'Excellent joke!'],
            'negative_feedback': ['1', 'Not funny at all'],
            'neutral_feedback': ['3', ''],
            'skip_feedback': ['s'],
            'skip_with_lowercase': ['s'],
            'invalid_then_valid': ['invalid', '4', 'Good after correction'],
            'empty_then_valid': ['', '3', 'Finally answered'],
            'rating_only': ['4', ''],
            'keyboard_interrupt': KeyboardInterrupt(),
            'eof_error': EOFError()
        }
    
    @staticmethod
    def create_cli_argument_scenarios() -> List[Dict[str, Any]]:
        """Create various CLI argument scenarios for testing."""
        return [
            {
                'name': 'no_arguments',
                'args': [],
                'expected_category': None,
                'expected_profile': None,
                'expected_no_feedback': False,
                'expected_stats': False
            },
            {
                'name': 'category_only',
                'args': ['--category', 'programming'],
                'expected_category': 'programming',
                'expected_profile': None,
                'expected_no_feedback': False,
                'expected_stats': False
            },
            {
                'name': 'profile_only',
                'args': ['--profile', 'test-profile'],
                'expected_category': None,
                'expected_profile': 'test-profile',
                'expected_no_feedback': False,
                'expected_stats': False
            },
            {
                'name': 'no_feedback_flag',
                'args': ['--no-feedback'],
                'expected_category': None,
                'expected_profile': None,
                'expected_no_feedback': True,
                'expected_stats': False
            },
            {
                'name': 'stats_flag',
                'args': ['--stats'],
                'expected_category': None,
                'expected_profile': None,
                'expected_no_feedback': False,
                'expected_stats': True
            },
            {
                'name': 'category_and_profile',
                'args': ['--category', 'dad-jokes', '--profile', 'my-profile'],
                'expected_category': 'dad-jokes',
                'expected_profile': 'my-profile',
                'expected_no_feedback': False,
                'expected_stats': False
            },
            {
                'name': 'category_and_no_feedback',
                'args': ['--category', 'puns', '--no-feedback'],
                'expected_category': 'puns',
                'expected_profile': None,
                'expected_no_feedback': True,
                'expected_stats': False
            }
        ]


class MockEnvironments:
    """Mock environment configurations for testing."""
    
    @staticmethod
    def create_aws_environment() -> Dict[str, str]:
        """Create AWS environment variables for testing."""
        return {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'AWS_DEFAULT_REGION': 'us-east-1',
            'AWS_PROFILE': 'test-profile'
        }
    
    @staticmethod
    def create_debug_environment() -> Dict[str, str]:
        """Create debug environment for testing."""
        return {
            'JOKE_CLI_DEBUG': 'true',
            'JOKE_CLI_LOG_LEVEL': 'DEBUG'
        }
    
    @staticmethod
    def create_empty_environment() -> Dict[str, str]:
        """Create empty environment for testing credential errors."""
        return {}


# Convenience functions for common test scenarios
def create_successful_joke_workflow_mocks():
    """Create mocks for a successful joke generation workflow."""
    return {
        'bedrock_client': Mock(**{
            'invoke_model.return_value': MockJokes.PROGRAMMING_JOKES[0]
        }),
        'feedback_storage': Mock(**{
            'save_feedback.return_value': True,
            'get_statistics.return_value': MockFeedbackData.create_feedback_statistics()
        }),
        'user_input': MockUserInput.create_feedback_inputs()['positive_feedback']
    }


def create_error_scenario_mocks(error_type: str = 'access_denied'):
    """Create mocks for error scenarios."""
    from botocore.exceptions import ClientError
    
    error_responses = MockErrorResponses.get_common_aws_errors()
    error_response = error_responses.get(error_type, error_responses['access_denied'])
    
    return {
        'bedrock_client': Mock(**{
            'invoke_model.side_effect': ClientError(error_response, 'InvokeModel')
        }),
        'feedback_storage': Mock(),
        'error_response': error_response
    }