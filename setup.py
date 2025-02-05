from setuptools import setup, find_packages

# Base requirements without version-sensitive packages
BASE_REQUIREMENTS = [
    "uvicorn>=0.15.0",
    "python-dotenv>=0.19.0",
    "pyyaml>=6.0.0",
    "dspy-ai>=2.0.0",
    "prometheus-client>=0.17.0",
    "typing-extensions>=4.0.0",
]

# Main app requirements that might conflict with Modal
STANDARD_REQUIREMENTS = [
    "fastapi>=0.115.8",
    "pydantic>=2.0.0,<3.0.0",
    "pydantic-settings>=2.0.0",
    "rich>=13.7.1",
    "importlib-metadata>=6.8.0",
    "typer>=0.12.3",
    "cloudpickle>=3.1.1",
]

# Modal's requirements (carefully versioned to match Modal's needs)
MODAL_REQUIREMENTS = [
    "fastapi==0.88.0",
    "pydantic>=1.10.0,<2.0.0",
    "pydantic-settings<2.0.0",  # Must be compatible with pydantic v1
    "rich==12.3.0",
    "importlib-metadata==4.8.1",
    "typer>=0.9",
    "cloudpickle==2.0.0",
    "aiohttp==3.8.3",
    "aiostream==0.4.4",
    "asgiref==3.5.2",
    "grpclib==0.4.7",
    "tblib==1.7.0",
    "ddtrace==1.5.2",
    "fastprogress==1.0.0",
]

setup(
    name="llm-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=BASE_REQUIREMENTS,
    extras_require={
        'standard': STANDARD_REQUIREMENTS,
        'modal': MODAL_REQUIREMENTS,
        'dev': [
            'pytest>=7.0.0',
            'python-multipart>=0.0.5',
            'httpx>=0.23.0',
            'email-validator>=2.0.0',
        ],
    },
    python_requires='>=3.9',
    description="A lightweight, extensible server for working with large language models",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Charles Feinn",
    author_email="charles@appsimple.io",
    url="https://github.com/chuckfinca/llm-server",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)