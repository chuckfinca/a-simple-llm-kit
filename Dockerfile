# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, our fast package installer
RUN pip install uv

# Copy all the files required for the build process first.
# This includes the project definition, the readme, and the source code itself.
COPY pyproject.toml ./
COPY README.md ./
COPY ./src ./src
COPY ./run.py ./

# Now that all source files are present, install the project and its dependencies.
# The `.` tells uv to install the local project found in the current directory.
RUN uv pip install --system .

# Copy the application configuration
COPY ./config ./config

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application.
CMD ["python", "run.py"]