"""
Unit tests for data models in the Joke CLI application.

Tests validation, serialization, and business logic for all data model classes.
"""

import pytest
from datetime import datetime
from uuid import uuid4
import uuid

from joke_cli.models import JokeRequest, JokeResponse, FeedbackEntry, BedrockConfig


class TestJokeRequest:
    """Test cases for JokeRequest data model."""
    
    def test_valid_joke_request_default(self):
        """Test creating a valid JokeRequest with default values."""
        request = JokeRequest()
        
        assert request.category is None
        assert request.aws_profile is None
        assert request.model_id == "amazon.titan-text-express-v1"
    
    def test_valid_joke_request_with_category(self):
        """Test creating a valid JokeRequest with a category."""
        request = JokeRequest(category="programming")
        
        assert request.category == "programming"
        assert request.aws_profile is None
        assert request.model_id == "amazon.titan-text-express-v1"
    
    def test_valid_joke_request_all_fields(self):
        """Test creating a valid JokeRequest with all fields."""
        request = JokeRequest(
            category="dad-jokes",
            aws_profile="test-profile",
            model_id="custom-model"
        )
        
        assert request.category == "dad-jokes"
        assert request.aws_profile == "test-profile"
        assert request.model_id == "custom-model"
    
    def test_invalid_category(self):
        """Test that invalid categories raise ValueError."""
        with pytest.raises(ValueError, match="Invalid category 'invalid'"):
            JokeRequest(category="invalid")
    
    def test_valid_categories(self):
        """Test that all valid categories are accepted."""
        valid_categories = ["general", "programming", "dad-jokes", "puns", "clean"]
        
        for category in valid_categories:
            request = JokeRequest(category=category)
            assert request.category == category
    
    def test_empty_model_id(self):
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            JokeRequest(model_id="")
    
    def test_none_model_id(self):
        """Test that None model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            JokeRequest(model_id=None)
    
    def test_invalid_aws_profile_type(self):
        """Test that non-string aws_profile raises ValueError."""
        with pytest.raises(ValueError, match="aws_profile must be a string"):
            JokeRequest(aws_profile=123)


class TestJokeResponse:
    """Test cases for JokeResponse data model."""
    
    def test_create_success_response(self):
        """Test creating a successful JokeResponse."""
        response = JokeResponse.create_success("Why did the chicken cross the road?", "general")
        
        assert response.success is True
        assert response.joke_text == "Why did the chicken cross the road?"
        assert response.category == "general"
        assert response.error_message is None
        assert isinstance(response.timestamp, datetime)
        assert isinstance(uuid.UUID(response.joke_id), uuid.UUID)
    
    def test_create_error_response(self):
        """Test creating an error JokeResponse."""
        response = JokeResponse.create_error("API Error", "programming")
        
        assert response.success is False
        assert response.joke_text == ""
        assert response.category == "programming"
        assert response.error_message == "API Error"
        assert isinstance(response.timestamp, datetime)
        assert isinstance(uuid.UUID(response.joke_id), uuid.UUID)
    
    def test_create_error_response_default_category(self):
        """Test creating an error JokeResponse with default category."""
        response = JokeResponse.create_error("Network Error")
        
        assert response.success is False
        assert response.category == "unknown"
        assert response.error_message == "Network Error"
    
    def test_valid_manual_creation(self):
        """Test manually creating a valid JokeResponse."""
        joke_id = str(uuid4())
        timestamp = datetime.now()
        
        response = JokeResponse(
            joke_id=joke_id,
            joke_text="Test joke",
            category="test",
            success=True,
            timestamp=timestamp
        )
        
        assert response.joke_id == joke_id
        assert response.joke_text == "Test joke"
        assert response.category == "test"
        assert response.success is True
        assert response.timestamp == timestamp
        assert response.error_message is None
    
    def test_invalid_joke_id_empty(self):
        """Test that empty joke_id raises ValueError."""
        with pytest.raises(ValueError, match="joke_id must be a non-empty string"):
            JokeResponse(
                joke_id="",
                joke_text="Test",
                category="test",
                success=True,
                timestamp=datetime.now()
            )
    
    def test_invalid_joke_id_format(self):
        """Test that invalid UUID format raises ValueError."""
        with pytest.raises(ValueError, match="joke_id must be a valid UUID string"):
            JokeResponse(
                joke_id="not-a-uuid",
                joke_text="Test",
                category="test",
                success=True,
                timestamp=datetime.now()
            )
    
    def test_invalid_joke_text_type(self):
        """Test that non-string joke_text raises ValueError."""
        with pytest.raises(ValueError, match="joke_text must be a string"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text=123,
                category="test",
                success=True,
                timestamp=datetime.now()
            )
    
    def test_invalid_category_empty(self):
        """Test that empty category raises ValueError."""
        with pytest.raises(ValueError, match="category must be a non-empty string"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="",
                success=True,
                timestamp=datetime.now()
            )
    
    def test_invalid_success_type(self):
        """Test that non-boolean success raises ValueError."""
        with pytest.raises(ValueError, match="success must be a boolean"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                success="true",
                timestamp=datetime.now()
            )
    
    def test_invalid_timestamp_type(self):
        """Test that non-datetime timestamp raises ValueError."""
        with pytest.raises(ValueError, match="timestamp must be a datetime object"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                success=True,
                timestamp="2024-01-01"
            )
    
    def test_success_without_joke_text(self):
        """Test that successful response without joke text raises ValueError."""
        with pytest.raises(ValueError, match="Successful responses must have non-empty joke_text"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text="",
                category="test",
                success=True,
                timestamp=datetime.now()
            )
    
    def test_failure_without_error_message(self):
        """Test that failed response without error message raises ValueError."""
        with pytest.raises(ValueError, match="Failed responses must have an error_message"):
            JokeResponse(
                joke_id=str(uuid4()),
                joke_text="",
                category="test",
                success=False,
                timestamp=datetime.now()
            )


class TestFeedbackEntry:
    """Test cases for FeedbackEntry data model."""
    
    def test_create_feedback_entry(self):
        """Test creating a FeedbackEntry using the create class method."""
        joke_id = str(uuid4())
        feedback = FeedbackEntry.create(
            joke_id=joke_id,
            joke_text="Test joke",
            category="general",
            rating=4,
            user_comment="Pretty funny!"
        )
        
        assert feedback.joke_id == joke_id
        assert feedback.joke_text == "Test joke"
        assert feedback.category == "general"
        assert feedback.rating == 4
        assert feedback.user_comment == "Pretty funny!"
        assert isinstance(feedback.timestamp, datetime)
    
    def test_create_feedback_entry_no_comment(self):
        """Test creating a FeedbackEntry without user comment."""
        joke_id = str(uuid4())
        feedback = FeedbackEntry.create(
            joke_id=joke_id,
            joke_text="Test joke",
            category="programming",
            rating=3
        )
        
        assert feedback.joke_id == joke_id
        assert feedback.rating == 3
        assert feedback.user_comment is None
    
    def test_valid_manual_creation(self):
        """Test manually creating a valid FeedbackEntry."""
        joke_id = str(uuid4())
        timestamp = datetime.now()
        
        feedback = FeedbackEntry(
            joke_id=joke_id,
            joke_text="Manual joke",
            category="puns",
            rating=5,
            timestamp=timestamp,
            user_comment="Excellent!"
        )
        
        assert feedback.joke_id == joke_id
        assert feedback.joke_text == "Manual joke"
        assert feedback.category == "puns"
        assert feedback.rating == 5
        assert feedback.timestamp == timestamp
        assert feedback.user_comment == "Excellent!"
    
    def test_invalid_joke_id_empty(self):
        """Test that empty joke_id raises ValueError."""
        with pytest.raises(ValueError, match="joke_id must be a non-empty string"):
            FeedbackEntry(
                joke_id="",
                joke_text="Test",
                category="test",
                rating=3,
                timestamp=datetime.now()
            )
    
    def test_invalid_joke_id_format(self):
        """Test that invalid UUID format raises ValueError."""
        with pytest.raises(ValueError, match="joke_id must be a valid UUID string"):
            FeedbackEntry(
                joke_id="invalid-uuid",
                joke_text="Test",
                category="test",
                rating=3,
                timestamp=datetime.now()
            )
    
    def test_invalid_rating_type(self):
        """Test that non-integer rating raises ValueError."""
        with pytest.raises(ValueError, match="rating must be an integer"):
            FeedbackEntry(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                rating="3",
                timestamp=datetime.now()
            )
    
    def test_invalid_rating_too_low(self):
        """Test that rating below 1 raises ValueError."""
        with pytest.raises(ValueError, match="rating must be between 1 and 5"):
            FeedbackEntry(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                rating=0,
                timestamp=datetime.now()
            )
    
    def test_invalid_rating_too_high(self):
        """Test that rating above 5 raises ValueError."""
        with pytest.raises(ValueError, match="rating must be between 1 and 5"):
            FeedbackEntry(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                rating=6,
                timestamp=datetime.now()
            )
    
    def test_valid_ratings(self):
        """Test that all valid ratings (1-5) are accepted."""
        joke_id = str(uuid4())
        
        for rating in range(1, 6):
            feedback = FeedbackEntry(
                joke_id=joke_id,
                joke_text="Test",
                category="test",
                rating=rating,
                timestamp=datetime.now()
            )
            assert feedback.rating == rating
    
    def test_invalid_user_comment_type(self):
        """Test that non-string user_comment raises ValueError."""
        with pytest.raises(ValueError, match="user_comment must be a string or None"):
            FeedbackEntry(
                joke_id=str(uuid4()),
                joke_text="Test",
                category="test",
                rating=3,
                timestamp=datetime.now(),
                user_comment=123
            )


class TestBedrockConfig:
    """Test cases for BedrockConfig data model."""
    
    def test_valid_bedrock_config_default(self):
        """Test creating a valid BedrockConfig with default values."""
        config = BedrockConfig(model_id="amazon.titan-text-express-v1")
        
        assert config.model_id == "amazon.titan-text-express-v1"
        assert config.max_tokens == 200
        assert config.temperature == 0.7
        assert config.top_p == 0.9
    
    def test_valid_bedrock_config_custom(self):
        """Test creating a valid BedrockConfig with custom values."""
        config = BedrockConfig(
            model_id="custom-model",
            max_tokens=500,
            temperature=0.5,
            top_p=0.8
        )
        
        assert config.model_id == "custom-model"
        assert config.max_tokens == 500
        assert config.temperature == 0.5
        assert config.top_p == 0.8
    
    def test_to_dict_conversion(self):
        """Test converting BedrockConfig to dictionary."""
        config = BedrockConfig(
            model_id="test-model",
            max_tokens=300,
            temperature=0.6,
            top_p=0.85
        )
        
        result = config.to_dict()
        expected = {
            "modelId": "test-model",
            "maxTokens": 300,
            "temperature": 0.6,
            "topP": 0.85
        }
        
        assert result == expected
    
    def test_invalid_model_id_empty(self):
        """Test that empty model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            BedrockConfig(model_id="")
    
    def test_invalid_model_id_none(self):
        """Test that None model_id raises ValueError."""
        with pytest.raises(ValueError, match="model_id must be a non-empty string"):
            BedrockConfig(model_id=None)
    
    def test_invalid_max_tokens_type(self):
        """Test that non-integer max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            BedrockConfig(model_id="test", max_tokens="200")
    
    def test_invalid_max_tokens_negative(self):
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BedrockConfig(model_id="test", max_tokens=-1)
    
    def test_invalid_max_tokens_zero(self):
        """Test that zero max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BedrockConfig(model_id="test", max_tokens=0)
    
    def test_invalid_max_tokens_too_high(self):
        """Test that max_tokens above limit raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be 4000 or less"):
            BedrockConfig(model_id="test", max_tokens=5000)
    
    def test_invalid_temperature_type(self):
        """Test that non-numeric temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be a number"):
            BedrockConfig(model_id="test", temperature="0.7")
    
    def test_invalid_temperature_too_low(self):
        """Test that temperature below 0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 1.0"):
            BedrockConfig(model_id="test", temperature=-0.1)
    
    def test_invalid_temperature_too_high(self):
        """Test that temperature above 1 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 1.0"):
            BedrockConfig(model_id="test", temperature=1.1)
    
    def test_valid_temperature_boundaries(self):
        """Test that temperature boundaries (0.0 and 1.0) are valid."""
        config1 = BedrockConfig(model_id="test", temperature=0.0)
        config2 = BedrockConfig(model_id="test", temperature=1.0)
        
        assert config1.temperature == 0.0
        assert config2.temperature == 1.0
    
    def test_invalid_top_p_type(self):
        """Test that non-numeric top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be a number"):
            BedrockConfig(model_id="test", top_p="0.9")
    
    def test_invalid_top_p_too_low(self):
        """Test that top_p below 0 raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            BedrockConfig(model_id="test", top_p=-0.1)
    
    def test_invalid_top_p_too_high(self):
        """Test that top_p above 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            BedrockConfig(model_id="test", top_p=1.1)
    
    def test_valid_top_p_boundaries(self):
        """Test that top_p boundaries (0.0 and 1.0) are valid."""
        config1 = BedrockConfig(model_id="test", top_p=0.0)
        config2 = BedrockConfig(model_id="test", top_p=1.0)
        
        assert config1.top_p == 0.0
        assert config2.top_p == 1.0