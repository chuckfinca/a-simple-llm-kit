import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Deploy LLM Server to Modal')
    parser.add_argument(
        '--name',
        required=True,
        help='App name (must be llm-server-{staging|production})'
    )
    args = parser.parse_args()

    try:
        # Validate app name
        if not args.name.startswith('llm-server-'):
            raise ValueError("App name must start with 'llm-server-'")
        environment = args.name.split('llm-server-')[1]
        if environment not in ("staging", "production"):
            raise ValueError("Environment must be either 'staging' or 'production'")
        
        # Get and deploy app
        from modal_app import get_app
        app = get_app(args.name)
        app.deploy()
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()