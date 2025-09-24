"""
Unit tests for the prompts module.

Tests prompt generation, category selection, and validation functionality.
"""

import pytest
from unittest.mock import patch

from joke_cli.prompts import (
    get_joke_prompt,
    get_random_category,
    get_available_categories,
    validate_category,
    JOKE_PROMPTS
)
from joke_cli.config import AVAILABLE_CATEGORIES


class TestGetJokePrompt:
    """Test cases for get_joke_prompt function."""
    
    def test_get_prompt_for_valid_category(self):
        """Test getting prompt for each valid category."""
        for category in AVAILABLE_CATEGORIES:
            prompt = get_joke_prompt(category)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert prompt == JOKE_PROMPTS[category]
    
    def test_get_prompt_for_general_category(self):
        """Test getting prompt for general category specifically."""
        prompt = get_joke_prompt("general")
        assert "clean, family-friendly joke" in prompt
        assert "appropriate for all audiences" in prompt
        assert "just the joke text" in prompt
    
    def test_get_prompt_for_programming_category(self):
        """Test getting prompt for programming category specifically."""
        prompt = get_joke_prompt("programming")
        assert "programming or computer science" in prompt
        assert "developers would appreciate" in prompt
        assert "just the joke text" in prompt
    
    def test_get_prompt_for_dad_jokes_category(self):
        """Test getting prompt for dad-jokes category specifically."""
        prompt = get_joke_prompt("dad-jokes")
        assert "classic dad joke" in prompt
        assert "groan and laugh" in prompt
        assert "just the joke text" in prompt
    
    def test_get_prompt_for_puns_category(self):
        """Test getting prompt for puns category specifically."""
        prompt = get_joke_prompt("puns")
        assert "pun-based joke" in prompt
        assert "wordplay" in prompt
        assert "just the joke text" in prompt
    
    def test_get_prompt_for_clean_category(self):
        """Test getting prompt for clean category specifically."""
        prompt = get_joke_prompt("clean")
        assert "wholesome, clean joke" in prompt
        assert "appropriate for children" in prompt
        assert "just the joke text" in prompt
    
    def test_get_prompt_for_invalid_category(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_joke_prompt("invalid-category")
        
        error_message = str(exc_info.value)
        assert "Invalid category 'invalid-category'" in error_message
        assert "Available categories:" in error_message
        for category in AVAILABLE_CATEGORIES:
            assert category in error_message
    
    @patch('joke_cli.prompts.get_random_category')
    def test_get_prompt_with_none_category_uses_random(self, mock_random):
        """Test that None category uses random category selection."""
        mock_random.return_value = "programming"
        
        prompt = get_joke_prompt(None)
        
        mock_random.assert_called_once()
        assert prompt == JOKE_PROMPTS["programming"]
    
    def test_get_prompt_with_none_category_returns_valid_prompt(self):
        """Test that None category returns a valid prompt."""
        prompt = get_joke_prompt(None)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert prompt in JOKE_PROMPTS.values()


class TestGetRandomCategory:
    """Test cases for get_random_category function."""
    
    def test_returns_valid_category(self):
        """Test that random category is always valid."""
        for _ in range(10):  # Test multiple times due to randomness
            category = get_random_category()
            assert category in AVAILABLE_CATEGORIES
    
    def test_returns_string(self):
        """Test that random category returns a string."""
        category = get_random_category()
        assert isinstance(category, str)
    
    @patch('random.choice')
    def test_uses_random_choice(self, mock_choice):
        """Test that function uses random.choice with available categories."""
        mock_choice.return_value = "general"
        
        result = get_random_category()
        
        mock_choice.assert_called_once_with(AVAILABLE_CATEGORIES)
        assert result == "general"
    
    def test_can_return_all_categories(self):
        """Test that all categories can potentially be returned."""
        # This is a probabilistic test - run many times to increase confidence
        returned_categories = set()
        for _ in range(100):
            category = get_random_category()
            returned_categories.add(category)
        
        # We should see most or all categories in 100 attempts
        assert len(returned_categories) >= 3  # At least 3 out of 5 categories


class TestGetAvailableCategories:
    """Test cases for get_available_categories function."""
    
    def test_returns_all_categories(self):
        """Test that all available categories are returned."""
        categories = get_available_categories()
        assert categories == AVAILABLE_CATEGORIES
    
    def test_returns_copy_not_reference(self):
        """Test that function returns a copy, not the original list."""
        categories = get_available_categories()
        categories.append("new-category")
        
        # Original should be unchanged
        assert "new-category" not in AVAILABLE_CATEGORIES
        assert "new-category" not in get_available_categories()
    
    def test_returns_list(self):
        """Test that function returns a list."""
        categories = get_available_categories()
        assert isinstance(categories, list)
    
    def test_contains_expected_categories(self):
        """Test that returned list contains expected categories."""
        categories = get_available_categories()
        expected_categories = ["general", "programming", "dad-jokes", "puns", "clean"]
        
        for expected in expected_categories:
            assert expected in categories


class TestValidateCategory:
    """Test cases for validate_category function."""
    
    def test_valid_categories_return_true(self):
        """Test that all valid categories return True."""
        for category in AVAILABLE_CATEGORIES:
            assert validate_category(category) is True
    
    def test_invalid_categories_return_false(self):
        """Test that invalid categories return False."""
        invalid_categories = [
            "invalid",
            "not-a-category", 
            "GENERAL",  # Case sensitive
            "programming-jokes",
            "",
            "general ",  # With space
            " general"   # With leading space
        ]
        
        for invalid_category in invalid_categories:
            assert validate_category(invalid_category) is False
    
    def test_case_sensitive_validation(self):
        """Test that validation is case sensitive."""
        assert validate_category("general") is True
        assert validate_category("General") is False
        assert validate_category("GENERAL") is False
        assert validate_category("GeNeRaL") is False


class TestJokePromptsConstant:
    """Test cases for JOKE_PROMPTS constant."""
    
    def test_all_categories_have_prompts(self):
        """Test that all available categories have corresponding prompts."""
        for category in AVAILABLE_CATEGORIES:
            assert category in JOKE_PROMPTS
    
    def test_no_extra_prompts(self):
        """Test that there are no extra prompts for non-existent categories."""
        for category in JOKE_PROMPTS:
            assert category in AVAILABLE_CATEGORIES
    
    def test_prompts_are_non_empty_strings(self):
        """Test that all prompts are non-empty strings."""
        for category, prompt in JOKE_PROMPTS.items():
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0
    
    def test_prompts_contain_instruction_text(self):
        """Test that all prompts contain instruction text."""
        for category, prompt in JOKE_PROMPTS.items():
            # All prompts should ask for just the joke text
            assert "just the joke text" in prompt
            # All prompts should avoid additional commentary
            assert "without any additional commentary" in prompt
    
    def test_prompts_are_appropriate_length(self):
        """Test that prompts are reasonably detailed but not excessive."""
        for category, prompt in JOKE_PROMPTS.items():
            # Should be detailed enough to provide good guidance
            assert len(prompt) > 100
            # But not excessively long
            assert len(prompt) < 1000