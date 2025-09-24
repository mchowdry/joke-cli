"""
Joke of the Day CLI Application

A Python command line application that provides users with a daily joke 
using Amazon Bedrock LLM APIs.
"""

__version__ = "1.0.0"
__author__ = "Joke CLI Team"
__description__ = "A CLI tool for generating jokes using AWS Bedrock"

# Import main CLI function for easy access
from .cli import main

__all__ = ["main"]