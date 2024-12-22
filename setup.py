from setuptools import setup, find_packages

setup(
    name="llm-server",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=0.19.0",
        "dspy-ai>=2.0.0",
        "pyyaml>=6.0.0",
        "python-multipart>=0.0.5"
    ],
)