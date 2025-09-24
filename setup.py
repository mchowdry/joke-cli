"""
Setup configuration for the Joke CLI application.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read version from __init__.py
version = "1.0.0"

setup(
    name="joke-cli",
    version=version,
    author="Joke CLI Team",
    author_email="team@joke-cli.com",
    description="A CLI tool for generating jokes using AWS Bedrock",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mchowdry/joke-cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Utilities",
        "Topic :: Entertainment/Games",
    ],
    python_requires=">=3.8",
    install_requires=[
        "boto3>=1.26.0",
        "botocore>=1.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "joke=joke_cli.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)