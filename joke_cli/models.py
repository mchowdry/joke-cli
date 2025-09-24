"""
Data models for the Joke CLI application.

This module contains dataclasses that represent the core data structures
used throughout the application for joke requests, responses, feedback,
and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid
import re


@dataclass
class JokeRequest:
    """Represents a request for joke generation."""
    
    category: Optional[str] = None
    aws_profile: Optional[str] = None
    model_id: str = "amazon.titan-text-express-v1"
    
    def __post_init__(self):
        """Validate the joke request after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the joke request data."""
        # Validate category if provided
        if self.category is not None:
            valid_categories = {"general", "programming", "dad-jokes", "puns", "clean"}
            if self.category not in valid_categories:
                raise ValueError(f"Invalid category '{self.category}'. Must be one of: {', '.join(sorted(valid_categories))}")
        
        # Validate model_id
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValueError("model_id must be a non-empty string")
        
        # Validate aws_profile if provided
        if self.aws_profile is not None and not isinstance(self.aws_profile, str):
            raise ValueError("aws_profile must be a string")


@dataclass
class JokeResponse:
    """Represents a response from joke generation."""
    
    joke_id: str
    joke_text: str
    category: str
    success: bool
    timestamp: datetime
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate the joke response after initialization."""
        self.validate()
    
    @classmethod
    def create_success(cls, joke_text: str, category: str) -> 'JokeResponse':
        """Create a successful joke response."""
        return cls(
            joke_id=str(uuid.uuid4()),
            joke_text=joke_text,
            category=category,
            success=True,
            timestamp=datetime.now(),
            error_message=None
        )
    
    @classmethod
    def create_error(cls, error_message: str, category: str = "unknown") -> 'JokeResponse':
        """Create an error joke response."""
        return cls(
            joke_id=str(uuid.uuid4()),
            joke_text="",
            category=category,
            success=False,
            timestamp=datetime.now(),
            error_message=error_message
        )
    
    def validate(self) -> None:
        """Validate the joke response data."""
        # Validate joke_id
        if not self.joke_id or not isinstance(self.joke_id, str):
            raise ValueError("joke_id must be a non-empty string")
        
        # Validate UUID format for joke_id
        try:
            uuid.UUID(self.joke_id)
        except ValueError:
            raise ValueError("joke_id must be a valid UUID string")
        
        # Validate joke_text
        if not isinstance(self.joke_text, str):
            raise ValueError("joke_text must be a string")
        
        # Validate category
        if not self.category or not isinstance(self.category, str):
            raise ValueError("category must be a non-empty string")
        
        # Validate success flag
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        # Validate error_message
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValueError("error_message must be a string or None")
        
        # Business logic validation
        if self.success and not self.joke_text.strip():
            raise ValueError("Successful responses must have non-empty joke_text")
        
        if not self.success and not self.error_message:
            raise ValueError("Failed responses must have an error_message")


@dataclass
class FeedbackEntry:
    """Represents user feedback for a joke."""
    
    joke_id: str
    joke_text: str
    category: str
    rating: int
    timestamp: datetime
    user_comment: Optional[str] = None
    
    def __post_init__(self):
        """Validate the feedback entry after initialization."""
        self.validate()
    
    @classmethod
    def create(cls, joke_id: str, joke_text: str, category: str, rating: int, 
               user_comment: Optional[str] = None) -> 'FeedbackEntry':
        """Create a new feedback entry with current timestamp."""
        return cls(
            joke_id=joke_id,
            joke_text=joke_text,
            category=category,
            rating=rating,
            timestamp=datetime.now(),
            user_comment=user_comment
        )
    
    def validate(self) -> None:
        """Validate the feedback entry data."""
        # Validate joke_id
        if not self.joke_id or not isinstance(self.joke_id, str):
            raise ValueError("joke_id must be a non-empty string")
        
        # Validate UUID format for joke_id
        try:
            uuid.UUID(self.joke_id)
        except ValueError:
            raise ValueError("joke_id must be a valid UUID string")
        
        # Validate joke_text
        if not isinstance(self.joke_text, str):
            raise ValueError("joke_text must be a string")
        
        # Validate category
        if not self.category or not isinstance(self.category, str):
            raise ValueError("category must be a non-empty string")
        
        # Validate rating (1-5 scale)
        if not isinstance(self.rating, int):
            raise ValueError("rating must be an integer")
        
        if not (1 <= self.rating <= 5):
            raise ValueError("rating must be between 1 and 5 (inclusive)")
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        # Validate user_comment
        if self.user_comment is not None and not isinstance(self.user_comment, str):
            raise ValueError("user_comment must be a string or None")


@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock API calls."""
    
    model_id: str
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        """Validate the Bedrock configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate the Bedrock configuration data."""
        # Validate model_id
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValueError("model_id must be a non-empty string")
        
        # Validate max_tokens
        if not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.max_tokens > 4000:  # Reasonable upper limit
            raise ValueError("max_tokens must be 4000 or less")
        
        # Validate temperature
        if not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number")
        
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0")
        
        # Validate top_p
        if not isinstance(self.top_p, (int, float)):
            raise ValueError("top_p must be a number")
        
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
    
    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary for API calls."""
        return {
            "modelId": self.model_id,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "topP": self.top_p
        }