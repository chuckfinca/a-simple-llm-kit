import sys
import modal
import argparse
from typing import Literal
from modal import volume
from modal_app import app  # Import the app instance

EnvironmentType = Literal["staging", "production"] 

def configure_for_environment(environment: EnvironmentType):
    """Configure app for specific environment"""
    app_name = f"llm-server-{environment}"
    
    # Update app name
    app.name = app_name
    
    # Configure volumes
    volume.name = f"{app_name}-logs"
    
    # Update secrets to use environment-specific ones
    app.fastapi_app.update(
        secrets=[modal.Secret.from_name(f"llm-server-{environment}-secrets")]
    )

def main():
    parser = argparse.ArgumentParser(description='Deploy LLM Server to Modal')
    parser.add_argument(
        '--name',
        required=True,
        help='App name (must be llm-server-{staging|production})'
    )
    args = parser.parse_args()

    try:
        # Extract and validate environment
        if not args.name.startswith('llm-server-'):
            raise ValueError("App name must start with 'llm-server-'")
        environment = args.name.split('llm-server-')[1]
        if environment not in ("staging", "production"):
            raise ValueError("Environment must be either 'staging' or 'production'")

        # Configure app for environment
        configure_for_environment(environment)
        
        # Deploy the app
        app.deploy()
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()