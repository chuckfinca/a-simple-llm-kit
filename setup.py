from setuptools import setup, find_packages

setup(
    name="llm-server",
    version="0.1.0",
    packages=find_packages(),
    extras_require={
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