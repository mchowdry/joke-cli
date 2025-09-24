"""
Joke generation service for the Joke CLI application.

This module contains the core business logic for joke generation,
category management, and feedback collection. It integrates with
the Bedrock client, prompt templates, and feedback storage.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .bedrock_client import BedrockClient, BedrockClientError, create_bedrock_client
from .models import JokeRequest, JokeResponse, FeedbackEntry, BedrockConfig
from .prompts import get_joke_prompt, get_available_categories, validate_category, get_random_category
from .feedback_storage import FeedbackStorage, get_default_storage
from .config import (
    DEFAULT_MODEL_ID,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FEEDBACK_RATING_MIN,
    FEEDBACK_RATING_MAX
)
from .error_handler import get_error_handler


logger = logging.getLogger(__name__)


class JokeServiceError(Exception):
    """Custom exception for joke service errors."""
    pass


class JokeService:
    """Core service for joke generation and feedback management."""
    
    def __init__(self, 
                 bedrock_client: Optional[BedrockClient] = None,
                 feedback_storage: Optional[FeedbackStorage] = None):
        """
        Initialize the joke service.
        
        Args:
            bedrock_client: Optional Bedrock client instance
            feedback_storage: Optional feedback storage instance
        """
        self._bedrock_client = bedrock_client
        self._feedback_storage = feedback_storage or get_default_storage()
    
    def _get_bedrock_client(self, profile: Optional[str] = None) -> BedrockClient:
        """Get or create a Bedrock client instance."""
        if self._bedrock_client is None:
            self._bedrock_client = create_bedrock_client(profile=profile)
        return self._bedrock_client
    
    def generate_joke(self, 
                     category: Optional[str] = None,
                     aws_profile: Optional[str] = None,
                     model_id: Optional[str] = None) -> JokeResponse:
        """
        Generate a joke for the specified category.
        
        Args:
            category: Joke category (if None, uses random category)
            aws_profile: AWS profile to use for Bedrock client
            model_id: Bedrock model ID to use
            
        Returns:
            JokeResponse object containing the generated joke or error information
        """
        try:
            # Validate and set category
            if category is None:
                category = get_random_category()
            elif not validate_category(category):
                error_handler = get_error_handler()
                available = ", ".join(get_available_categories())
                error_info = error_handler.format_error_message(
                    "invalid_category",
                    category=category,
                    available_categories=available
                )
                return JokeResponse.create_error(error_info["message"], category)
            
            # Set default model if not provided
            if model_id is None:
                model_id = DEFAULT_MODEL_ID
            
            logger.info(f"Generating joke for category: {category}, model: {model_id}")
            
            # Get the appropriate prompt for the category
            prompt = get_joke_prompt(category)
            
            # Create Bedrock configuration
            bedrock_config = BedrockConfig(
                model_id=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P
            )
            
            # Get Bedrock client and invoke model
            client = self._get_bedrock_client(profile=aws_profile)
            joke_text = client.invoke_model(prompt, bedrock_config)
            
            # Clean up the joke text
            cleaned_joke = self._clean_joke_text(joke_text)
            
            if not cleaned_joke.strip():
                error_handler = get_error_handler()
                error_info = error_handler.format_error_message("empty_response")
                return JokeResponse.create_error(error_info["message"], category)
            
            logger.info(f"Successfully generated joke of length: {len(cleaned_joke)}")
            return JokeResponse.create_success(cleaned_joke, category)
            
        except BedrockClientError as e:
            logger.error(f"Bedrock client error: {e}")
            return JokeResponse.create_error(str(e), category or "unknown")
        except Exception as e:
            logger.error(f"Unexpected error generating joke: {e}")
            return JokeResponse.create_error(f"Unexpected error: {e}", category or "unknown")
    
    def _clean_joke_text(self, joke_text: str) -> str:
        """
        Clean and format the generated joke text.
        
        Args:
            joke_text: Raw joke text from the model
            
        Returns:
            Cleaned and formatted joke text
        """
        # Remove common prefixes and suffixes that models might add
        prefixes_to_remove = [
            "Here's a joke for you:",
            "Here's a joke:",
            "Joke:",
            "Here you go:",
            "Sure, here's a joke:",
            "Here's one:",
        ]
        
        suffixes_to_remove = [
            "Hope you enjoyed it!",
            "Hope that made you smile!",
            "I hope you found that funny!",
            "Did you like it?",
        ]
        
        cleaned = joke_text.strip()
        
        # Remove prefixes (case insensitive)
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove suffixes (case insensitive)
        for suffix in suffixes_to_remove:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[:-len(suffix)].strip()
                break
        
        # Remove extra whitespace and normalize line breaks
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        return cleaned
    
    def format_joke_output(self, joke_response: JokeResponse) -> str:
        """
        Format a joke response for display to the user.
        
        Args:
            joke_response: The JokeResponse object to format
            
        Returns:
            Formatted string ready for display
        """
        if not joke_response.success:
            return f"Error: {joke_response.error_message}"
        
        # Format the joke with proper spacing and category info
        formatted_lines = [
            "🎭 Joke of the Day 🎭",
            "",
            joke_response.joke_text,
            "",
            f"Category: {joke_response.category.title()}"
        ]
        
        return "\n".join(formatted_lines)
    
    def collect_user_feedback(self, joke_response: JokeResponse, rating: int, 
                            user_comment: Optional[str] = None) -> bool:
        """
        Store user feedback for a joke.
        
        Args:
            joke_response: The JokeResponse that was rated
            rating: User rating (1-5 scale)
            user_comment: Optional user comment
            
        Returns:
            True if feedback was saved successfully, False otherwise
        """
        try:
            if not joke_response.success:
                logger.warning("Attempted to collect feedback for failed joke response")
                return False
            
            feedback = FeedbackEntry.create(
                joke_id=joke_response.joke_id,
                joke_text=joke_response.joke_text,
                category=joke_response.category,
                rating=rating,
                user_comment=user_comment
            )
            
            self._feedback_storage.save_feedback(feedback)
            logger.info(f"Saved feedback for joke {joke_response.joke_id}: rating={rating}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def prompt_for_feedback(self) -> tuple[Optional[int], Optional[str]]:
        """
        Prompt the user for feedback on a joke.
        
        Returns:
            Tuple of (rating, comment) where rating is 1-5 or None if skipped,
            and comment is optional user comment or None
        """
        error_handler = get_error_handler()
        
        try:
            # Prompt for rating
            while True:
                rating_input = input("\nHow would you rate this joke? (1-5, or 's' to skip): ").strip().lower()
                
                if rating_input == 's' or rating_input == 'skip':
                    return None, None
                
                try:
                    rating = int(rating_input)
                    if FEEDBACK_RATING_MIN <= rating <= FEEDBACK_RATING_MAX:
                        break
                    else:
                        error_info = error_handler.format_error_message("invalid_rating", rating=rating_input)
                        print(f"❌ {error_info['message']}")
                        for guidance in error_info['guidance']:
                            print(f"   {guidance}")
                except ValueError:
                    error_info = error_handler.format_error_message("invalid_rating", rating=rating_input)
                    print(f"❌ {error_info['message']}")
                    for guidance in error_info['guidance']:
                        print(f"   {guidance}")
            
            # Prompt for optional comment
            comment_input = input("Any comments? (optional, press Enter to skip): ").strip()
            comment = comment_input if comment_input else None
            
            return rating, comment
            
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or EOF gracefully
            print("\nFeedback skipped.")
            return None, None
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            error_handler.display_warning("Could not collect feedback due to an unexpected error")
            return None, None
    
    def get_available_categories(self) -> List[str]:
        """
        Get the list of available joke categories.
        
        Returns:
            List of available category names
        """
        return get_available_categories()
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.
        
        Returns:
            Dictionary containing feedback statistics
        """
        try:
            return self._feedback_storage.get_feedback_stats()
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {
                "total_jokes": 0,
                "average_rating": 0.0,
                "category_stats": {}
            }
    
    def format_statistics_output(self, stats: Optional[Dict[str, Any]] = None) -> str:
        """
        Format feedback statistics for display.
        
        Args:
            stats: Optional statistics dictionary (fetches if not provided)
            
        Returns:
            Formatted statistics string
        """
        if stats is None:
            stats = self.get_feedback_statistics()
        
        if stats["total_jokes"] == 0:
            return "📊 Feedback Statistics\n\nNo feedback data available yet. Generate some jokes and rate them!"
        
        lines = [
            "📊 Feedback Statistics",
            "=" * 40,
            f"📈 Total jokes rated: {stats['total_jokes']}",
            f"⭐ Average rating: {stats['average_rating']:.1f}/5.0",
            ""
        ]
        
        # Add rating distribution
        rating_dist = self._calculate_rating_distribution()
        if rating_dist:
            lines.append("📊 Rating Distribution:")
            lines.append("-" * 25)
            for rating in range(5, 0, -1):  # Show 5 to 1 stars
                count = rating_dist.get(rating, 0)
                percentage = (count / stats['total_jokes'] * 100) if stats['total_jokes'] > 0 else 0
                stars = "⭐" * rating
                bar = "█" * int(percentage / 5)  # Scale bar to fit
                lines.append(f"{stars} ({rating}): {count:2d} jokes {bar} {percentage:4.1f}%")
            lines.append("")
        
        # Add category breakdown if available
        category_stats = stats.get("category_stats", {})
        if category_stats:
            lines.append("📂 By Category:")
            lines.append("-" * 20)
            
            # Sort categories by count (descending)
            sorted_categories = sorted(
                category_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            for category, cat_stats in sorted_categories:
                count = cat_stats["count"]
                avg_rating = cat_stats["avg_rating"]
                percentage = (count / stats['total_jokes'] * 100) if stats['total_jokes'] > 0 else 0
                lines.append(f"  {category.title()}: {count} jokes ({percentage:.1f}%), {avg_rating:.1f}/5.0 avg")
            
            # Add most/least popular categories
            if len(sorted_categories) > 1:
                lines.append("")
                most_popular = sorted_categories[0]
                least_popular = sorted_categories[-1]
                lines.append(f"🏆 Most popular: {most_popular[0].title()} ({most_popular[1]['count']} jokes)")
                lines.append(f"📉 Least popular: {least_popular[0].title()} ({least_popular[1]['count']} jokes)")
                
                # Best and worst rated categories
                best_rated = max(sorted_categories, key=lambda x: x[1]["avg_rating"])
                worst_rated = min(sorted_categories, key=lambda x: x[1]["avg_rating"])
                if best_rated != worst_rated:  # Only show if different
                    lines.append(f"🌟 Highest rated: {best_rated[0].title()} ({best_rated[1]['avg_rating']:.1f}/5.0)")
                    lines.append(f"💭 Lowest rated: {worst_rated[0].title()} ({worst_rated[1]['avg_rating']:.1f}/5.0)")
        
        return "\n".join(lines)
    
    def _calculate_rating_distribution(self) -> Dict[int, int]:
        """
        Calculate the distribution of ratings.
        
        Returns:
            Dictionary mapping rating (1-5) to count
        """
        try:
            all_feedback = self._feedback_storage.get_all_feedback()
            distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            
            for feedback in all_feedback:
                if 1 <= feedback.rating <= 5:
                    distribution[feedback.rating] += 1
            
            return distribution
        except Exception as e:
            logger.error(f"Failed to calculate rating distribution: {e}")
            return {}
    
    def validate_joke_request(self, request: JokeRequest) -> Optional[str]:
        """
        Validate a joke request.
        
        Args:
            request: The JokeRequest to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        try:
            request.validate()
            return None
        except ValueError as e:
            return str(e)


# Convenience functions for module-level access
_default_service = None

def get_default_service() -> JokeService:
    """Get the default joke service instance."""
    global _default_service
    if _default_service is None:
        _default_service = JokeService()
    return _default_service

def generate_joke(category: Optional[str] = None,
                 aws_profile: Optional[str] = None,
                 model_id: Optional[str] = None) -> JokeResponse:
    """Generate a joke using the default service instance."""
    service = get_default_service()
    return service.generate_joke(category, aws_profile, model_id)

def collect_feedback(joke_response: JokeResponse, rating: int, 
                    user_comment: Optional[str] = None) -> bool:
    """Collect feedback using the default service instance."""
    service = get_default_service()
    return service.collect_user_feedback(joke_response, rating, user_comment)

def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics using the default service instance."""
    service = get_default_service()
    return service.get_feedback_statistics()

def format_joke_for_display(joke_response: JokeResponse) -> str:
    """Format a joke for display using the default service instance."""
    service = get_default_service()
    return service.format_joke_output(joke_response)

def format_stats_for_display(stats: Optional[Dict[str, Any]] = None) -> str:
    """Format statistics for display using the default service instance."""
    service = get_default_service()
    return service.format_statistics_output(stats)