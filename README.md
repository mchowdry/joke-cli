# Joke of the Day CLI

A Python command line application that generates entertaining jokes using Amazon Bedrock's advanced AI models. Get your daily dose of humor with AI-powered jokes across multiple categories!

## ✨ Features

- **AI-Powered Jokes**: Generate fresh, unique jokes using AWS Bedrock's Claude Sonnet 4 model
- **Multiple Categories**: Choose from 5 different joke categories:
  - `general` - Universal humor for everyone
  - `programming` - Tech and coding jokes for developers
  - `dad-jokes` - Classic dad jokes and puns
  - `puns` - Wordplay and pun-based humor
  - `clean` - Family-friendly, appropriate content
- **Interactive Feedback System**: Rate jokes (1-5 stars) and add optional comments
- **Statistics Dashboard**: View detailed feedback analytics and trends
- **AWS Integration**: Seamless integration with AWS Bedrock and credential management
- **Modern API Support**: Uses both legacy invoke_model and modern Converse APIs
- **Comprehensive Error Handling**: User-friendly error messages with actionable guidance

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd joke-of-the-day

# Install in development mode
pip install -e .
```

### Using pip (when published)

```bash
pip install joke-cli
```

## 📖 Usage

### Basic Commands

```bash
# Generate a random joke from any category
joke

# Generate a joke from a specific category
joke --category programming
joke -c dad-jokes

# Skip the feedback prompt
joke --no-feedback

# View comprehensive feedback statistics
joke --stats
joke -s

# Use a specific AWS profile
joke --profile my-aws-profile
joke -p production

# Display help and available options
joke --help

# Show version information
joke --version
```

### Example Output

```
🎭 Joke of the Day 🎭

Why do programmers prefer dark mode?
Because light attracts bugs.

Category: Programming

How would you rate this joke? (1-5, or 's' to skip): 4
Any comments? (optional, press Enter to skip): Pretty clever!
Thanks for your feedback!
```

### Statistics Dashboard

```bash
joke --stats
```

```
📊 Feedback Statistics
========================================
📈 Total jokes rated: 15
⭐ Average rating: 3.8/5.0

📊 Rating Distribution:
-------------------------
⭐⭐⭐⭐⭐ (5):  3 jokes ████████ 20.0%
⭐⭐⭐⭐ (4):  8 jokes ████████████████████ 53.3%
⭐⭐⭐ (3):  3 jokes ████████ 20.0%
⭐⭐ (2):  1 jokes ██ 6.7%
⭐ (1):  0 jokes   0.0%

📂 By Category:
--------------------
  Programming: 6 jokes (40.0%), 4.2/5.0 avg
  General: 5 jokes (33.3%), 3.6/5.0 avg
  Dad-jokes: 4 jokes (26.7%), 3.5/5.0 avg

🏆 Most popular: Programming (6 jokes)
🌟 Highest rated: Programming (4.2/5.0)
```

## 🔧 Requirements

- **Python**: 3.8 or higher (supports 3.8, 3.9, 3.10, 3.11)
- **AWS Account**: Active AWS account with Bedrock access
- **AWS Credentials**: Configured via one of:
  - `aws configure` (recommended)
  - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
  - IAM roles (for EC2/ECS/Lambda)
  - AWS profiles
- **Bedrock Access**: Access to Claude Sonnet 4 model in AWS Bedrock

## ⚙️ AWS Setup

### 1. Configure AWS Credentials

```bash
# Option 1: Use AWS CLI (recommended)
aws configure

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option 3: Use a specific profile
export AWS_PROFILE=my-profile
```

### 2. Request Bedrock Model Access

1. Open the [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to **Model access** in the left sidebar
3. Request access to **Claude Sonnet 4** model
4. Wait for approval (usually takes a few minutes)

### 3. Verify Setup

```bash
# Test your configuration
joke --category general
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or manually install dev requirements
pip install pytest pytest-mock black flake8 mypy
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=joke_cli

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

### Code Quality

```bash
# Format code
black joke_cli/

# Sort imports
isort joke_cli/

# Lint code
flake8 joke_cli/

# Type checking
mypy joke_cli/
```

### Project Structure

```
joke_cli/
├── __init__.py           # Package metadata
├── cli.py               # Command-line interface
├── bedrock_client.py    # AWS Bedrock integration
├── joke_service.py      # Core business logic
├── models.py            # Data models
├── config.py            # Configuration constants
├── feedback_storage.py  # Feedback persistence
├── prompts.py           # AI prompt templates
└── error_handler.py     # Error handling utilities

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── fixtures/            # Test data and mocks
```

## 🔍 Troubleshooting

### Common Issues

**"AWS credentials not found"**
```bash
# Configure credentials
aws configure
```

**"Access denied to Bedrock model"**
- Request model access in AWS Bedrock console
- Ensure IAM permissions include `bedrock:InvokeModel`

**"Model not supported on this API"**
- The app automatically handles different model APIs
- Ensure you're using a supported model version

### Getting Help

```bash
# View detailed help
joke --help

# Check version
joke --version
```

## 📊 Technical Details

- **Default Model**: `us.anthropic.claude-sonnet-4-20250514-v1:0`
- **API Support**: Both legacy `invoke_model` and modern `converse` APIs
- **Storage**: Local JSON file in `~/.joke_cli/joke_feedback.json`
- **Timeout**: 10 seconds for API calls
- **Retries**: Up to 3 attempts with adaptive retry strategy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Roadmap

- [ ] Support for additional AI models
- [ ] Custom joke prompts
- [ ] Export feedback data
- [ ] Joke sharing functionality
- [ ] Scheduled joke delivery