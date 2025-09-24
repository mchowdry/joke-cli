"""
Prompt template system for joke generation.

Contains category-specific prompts for AWS Bedrock joke generation
and functions to manage prompt selection and randomization.
"""

import random
from typing import Dict, Optional

from .config import AVAILABLE_CATEGORIES


# Category-specific joke generation prompts
JOKE_PROMPTS: Dict[str, str] = {
    "general": """Generate a clean, family-friendly joke that would be appropriate for all audiences. 
The joke should be clever, witty, and make people smile. Avoid any offensive content, 
controversial topics, or inappropriate language. Keep it light-hearted and fun.

Please provide just the joke text without any additional commentary or explanation.""",

    "programming": """Generate a programming or computer science related joke that developers would appreciate. 
The joke can reference coding concepts, programming languages, software development practices, 
debugging, or tech culture. Make it clever and relatable to people in the tech industry.
Keep it clean and professional.

Please provide just the joke text without any additional commentary or explanation.""",

    "dad-jokes": """Generate a classic dad joke - the kind that makes people groan and laugh at the same time. 
It should be a simple, punny, wholesome joke with a predictable but amusing punchline. 
Think of the type of joke a father might tell at a family dinner that gets eye rolls 
but secret smiles. Keep it clean and family-friendly.

Please provide just the joke text without any additional commentary or explanation.""",

    "puns": """Generate a clever pun-based joke that plays with words, double meanings, or similar sounds. 
The humor should come from wordplay, clever linguistic twists, or unexpected word associations. 
Make it witty and clever, the kind that makes people appreciate the creativity of language.
Keep it clean and appropriate for all audiences.

Please provide just the joke text without any additional commentary or explanation.""",

    "clean": """Generate a wholesome, clean joke that is completely appropriate for children and families. 
Avoid any adult themes, innuendo, or potentially offensive content. The joke should be 
innocent, sweet, and the kind you'd feel comfortable sharing with anyone. 
Focus on simple, cheerful humor that brings joy.

Please provide just the joke text without any additional commentary or explanation."""
}


def get_joke_prompt(category: Optional[str] = None) -> str:
    """
    Get a joke generation prompt for the specified category.
    
    Args:
        category: The joke category to get a prompt for. If None, returns a random category prompt.
        
    Returns:
        The prompt string for joke generation.
        
    Raises:
        ValueError: If the specified category is not supported.
    """
    if category is None:
        category = get_random_category()
    
    if category not in JOKE_PROMPTS:
        available = ", ".join(AVAILABLE_CATEGORIES)
        raise ValueError(f"Invalid category '{category}'. Available categories: {available}")
    
    return JOKE_PROMPTS[category]


def get_random_category() -> str:
    """
    Select a random joke category from available categories.
    
    Returns:
        A randomly selected category name.
    """
    return random.choice(AVAILABLE_CATEGORIES)


def get_available_categories() -> list[str]:
    """
    Get the list of all available joke categories.
    
    Returns:
        List of available category names.
    """
    return AVAILABLE_CATEGORIES.copy()


def validate_category(category: str) -> bool:
    """
    Validate if a category is supported.
    
    Args:
        category: The category name to validate.
        
    Returns:
        True if the category is valid, False otherwise.
    """
    return category in AVAILABLE_CATEGORIES