import os

import uvicorn


def main():
    """Entry point for the llm-server console script."""
    # Allows running with hot-reloading for development via an environment variable
    reload = os.getenv("LLM_SERVER_RELOAD", "false").lower() in ("true", "1", "t")
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=8000, reload=reload)


if __name__ == "__main__":
    main()
