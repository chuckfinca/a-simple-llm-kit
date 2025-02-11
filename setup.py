from setuptools import setup, find_packages

# Read requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return [
            line.strip() 
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Read README
with open('README.md') as f:
    long_description = f.read()

setup(
    name="llm-server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
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
    long_description=long_description,
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