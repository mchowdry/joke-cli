"""
Command-line interface for the Joke CLI application.

Handles argument parsing, user interaction, and application orchestration.
"""

import argparse
import sys
import os
from typing import Optional, List

from .config import (
    CLI_COMMAND_NAME,
    CLI_DESCRIPTION,
    AVAILABLE_CATEGORIES,
    EXIT_SUCCESS,
    EXIT_INVALID_ARGUMENTS,
    EXIT_USER_CANCELLED,
    DEFAULT_MODEL_ID
)
from .joke_service import JokeService, JokeServiceError
from .feedback_storage import FeedbackStorage
from .error_handler import get_error_handler, handle_error, display_error_message, validate_condition


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog=CLI_COMMAND_NAME,
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {CLI_COMMAND_NAME}                    Generate a random joke
  {CLI_COMMAND_NAME} --category programming    Generate a programming joke
  {CLI_COMMAND_NAME} --stats            Show feedback statistics
  {CLI_COMMAND_NAME} --no-feedback      Skip feedback collection

Available categories: {', '.join(AVAILABLE_CATEGORIES)}
        """
    )
    
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=AVAILABLE_CATEGORIES,
        help=f"Joke category to generate. Available: {', '.join(AVAILABLE_CATEGORIES)}"
    )
    
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        help="AWS profile to use for authentication"
    )
    
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Skip feedback collection after displaying the joke"
    )
    
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Display feedback statistics and exit"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"{CLI_COMMAND_NAME} 1.0.0"
    )
    
    return parser


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Optional list of arguments to parse (for testing)
        
    Returns:
        argparse.Namespace: Parsed arguments
        
    Raises:
        SystemExit: If argument parsing fails
    """
    parser = create_argument_parser()
    return parser.parse_args(args)


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments for consistency and correctness.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If arguments are invalid or inconsistent
    """
    error_handler = get_error_handler()
    
    # Check for mutually exclusive options
    if args.stats and args.category:
        error_handler.display_error(
            "Cannot specify --category with --stats option",
            ["Use either --stats to view statistics OR --category to generate a joke, not both."]
        )
        sys.exit(EXIT_INVALID_ARGUMENTS)
    
    if args.stats and args.no_feedback:
        error_handler.display_error(
            "Cannot specify --no-feedback with --stats option",
            ["The --stats option only displays statistics and doesn't generate jokes."]
        )
        sys.exit(EXIT_INVALID_ARGUMENTS)
    
    # Validate AWS profile if provided
    if args.profile:
        # Check if profile exists in AWS credentials
        try:
            import boto3
            session = boto3.Session(profile_name=args.profile)
            # Try to get credentials to validate profile exists
            session.get_credentials()
        except Exception as e:
            display_error_message(
                "invalid_profile", 
                profile=args.profile,
                exit_on_error=True
            )


def display_statistics() -> None:
    """Display feedback statistics to the user."""
    error_handler = get_error_handler()
    
    try:
        joke_service = JokeService()
        formatted_stats = joke_service.format_statistics_output()
        print(formatted_stats)
        
    except Exception as e:
        error_handler.handle_error(
            e, 
            error_code="feedback_storage_error",
            exit_on_error=True
        )


def initialize_application() -> None:
    """
    Initialize the application with proper setup and configuration.
    
    This function handles:
    - Environment variable validation
    - Logging configuration
    - System compatibility checks
    """
    import logging
    
    # Configure logging based on debug mode
    debug_mode = os.environ.get("JOKE_CLI_DEBUG", "").lower() in ("1", "true", "yes")
    log_level = logging.DEBUG if debug_mode else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    
    # Suppress boto3 debug logging unless explicitly requested
    if not debug_mode:
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)


def cleanup_application() -> None:
    """
    Perform cleanup operations before application exit.
    
    This function handles:
    - Flushing output streams
    - Cleanup of temporary resources
    """
    # Ensure all output is flushed
    sys.stdout.flush()
    sys.stderr.flush()


def orchestrate_joke_generation(parsed_args: argparse.Namespace) -> int:
    """
    Orchestrate the complete joke generation workflow.
    
    Args:
        parsed_args: Parsed command line arguments
        
    Returns:
        Exit code for the application
    """
    error_handler = get_error_handler()
    
    try:
        # Initialize joke service
        joke_service = JokeService()
        
        # Generate joke
        joke_response = joke_service.generate_joke(
            category=parsed_args.category,
            aws_profile=parsed_args.profile
        )
        
        # Check if joke generation was successful
        if not joke_response.success:
            error_handler.display_error(
                joke_response.error_message,
                ["Try running the command again or use a different category."]
            )
            return EXIT_SUCCESS  # Don't treat joke generation failure as app error
        
        # Display the joke with formatting
        formatted_joke = joke_service.format_joke_output(joke_response)
        print(formatted_joke)
        
        # Collect feedback unless disabled
        if not parsed_args.no_feedback:
            try:
                rating, comment = joke_service.prompt_for_feedback()
                if rating is not None:
                    success = joke_service.collect_user_feedback(joke_response, rating, comment)
                    if success:
                        print("Thanks for your feedback!")
                    else:
                        error_handler.display_warning("Could not save feedback, but the joke was generated successfully.")
            except KeyboardInterrupt:
                print("\nFeedback skipped.", file=sys.stderr)
                # Don't exit with error code for skipped feedback
            except Exception as e:
                error_handler.display_warning(f"Error collecting feedback: {e}")
                # Don't exit with error code for feedback issues
        
        return EXIT_SUCCESS
        
    except Exception as e:
        error_handler.handle_error(
            e,
            error_code="general_error",
            exit_on_error=False
        )
        return EXIT_SUCCESS  # Use general error code


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the CLI application.
    
    This function orchestrates the complete application flow:
    1. Application initialization and setup
    2. Command line argument parsing and validation
    3. Route to appropriate workflow (stats display or joke generation)
    4. Error handling and user feedback
    5. Application cleanup
    
    Args:
        args: Optional list of arguments (for testing)
    """
    exit_code = EXIT_SUCCESS
    
    try:
        # Initialize application
        initialize_application()
        
        # Initialize error handler with debug mode from environment
        debug_mode = os.environ.get("JOKE_CLI_DEBUG", "").lower() in ("1", "true", "yes")
        error_handler = get_error_handler(debug=debug_mode)
        
        # Parse and validate arguments
        parsed_args = parse_arguments(args)
        validate_arguments(parsed_args)
        
        # Route to appropriate workflow
        if parsed_args.stats:
            # Handle stats display workflow
            display_statistics()
        else:
            # Handle joke generation workflow
            exit_code = orchestrate_joke_generation(parsed_args)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        exit_code = EXIT_USER_CANCELLED
    except SystemExit as e:
        # Preserve exit codes from argument parsing or other system exits
        exit_code = e.code if e.code is not None else EXIT_SUCCESS
    except Exception as e:
        # Handle unexpected errors
        error_handler = get_error_handler()
        error_handler.handle_error(
            e,
            error_code="general_error",
            exit_on_error=False
        )
        exit_code = EXIT_SUCCESS
    finally:
        # Always perform cleanup
        cleanup_application()
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()