#!/usr/bin/env python
"""
Setup script for CNN-RNN Action Recognition project.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="cnn-rnn-action-recognition",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="CNN-RNN model for action recognition using video and skeleton data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cnn-rnn-action-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
        "notebook": [
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
    },
    entry_points={
        "console_scripts": [
            "action-recognition-train=scripts.train_model:main",
            "action-recognition-predict=scripts.predict:main",
            "action-recognition-features=scripts.generate_features:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt"],
    },
    zip_safe=False,
)
