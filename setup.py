"""
Setup configuration for Space Biology Knowledge Engine
"""

from setuptools import setup, find_packages

setup(
    name="space-biology-knowledge-engine",
    version="0.1.0",
    description="AI-powered space biology research assistant",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.2.16",
        "langchain-openai>=0.1.23",
        "langchain-community>=0.2.16",
        "langchain-huggingface>=0.0.3",
        "langgraph>=0.2.14",
        "qdrant-client>=1.10.1",
        "python-dotenv>=1.0.1",
        "pypdf>=4.3.1",
        "rich>=13.7.1"
    ],
    python_requires=">=3.9",
)
