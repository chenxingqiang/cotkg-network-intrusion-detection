#!/usr/bin/env python3
"""
Setup script for Ollama models.
This script helps users manage their local Ollama models.
"""

import os
import sys
import requests
import argparse
import logging
import time
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more information
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default Ollama settings
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODELS = ['llama2', 'mistral', 'codellama', 'vicuna']
MAX_RETRIES = 5  # Maximum number of retries for server connection
RETRY_DELAY = 2  # Delay between retries in seconds

# Configure requests to bypass proxy for localhost
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

def wait_for_server(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY) -> bool:
    """Wait for Ollama server to become available."""
    session = requests.Session()
    session.trust_env = False  # Disable proxy settings
    
    for i in range(max_retries):
        try:
            logger.debug(f"Attempting to connect to Ollama server (attempt {i+1}/{max_retries})")
            response = session.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama server: {response.json()}")
                return True
            else:
                logger.warning(f"Server responded with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Connection attempt {i+1} failed: {str(e)}")
        
        if i < max_retries - 1:  # Don't sleep on the last attempt
            logger.info(f"Waiting {delay} seconds before next attempt...")
            time.sleep(delay)
    
    return False

def get_session() -> requests.Session:
    """Create a session with proxy disabled."""
    session = requests.Session()
    session.trust_env = False
    return session

def check_ollama_server() -> bool:
    """Check if Ollama server is running."""
    session = get_session()
    try:
        # Try version endpoint first
        response = session.get(f"{OLLAMA_HOST}/api/version")
        if response.status_code == 200:
            logger.debug(f"Server version: {response.json()}")
            return True
            
        # Fallback to other endpoints if version fails
        response = session.get(OLLAMA_HOST)
        logger.debug(f"Server response status: {response.status_code}")
        return response.status_code != 404
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking server: {str(e)}")
        return False

def list_models() -> List[str]:
    """List all available models in Ollama."""
    session = get_session()
    try:
        response = session.get(f"{OLLAMA_HOST}/api/tags")
        logger.debug(f"List models response: {response.text}")
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        
        logger.warning(f"Failed to list models, status code: {response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama."""
    session = get_session()
    try:
        logger.info(f"Pulling model {model_name}...")
        response = session.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name}
        )
        logger.debug(f"Pull response: {response.text}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pulling model {model_name}: {str(e)}")
        return False

def delete_model(model_name: str) -> bool:
    """Delete a model from Ollama."""
    session = get_session()
    try:
        logger.info(f"Deleting model {model_name}...")
        response = session.delete(
            f"{OLLAMA_HOST}/api/delete",
            json={"name": model_name}
        )
        logger.debug(f"Delete response: {response.text}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting model {model_name}: {str(e)}")
        return False

def setup_models(models: Optional[List[str]] = None) -> None:
    """Set up specified models or default models."""
    if not wait_for_server():
        logger.error("Failed to connect to Ollama server after multiple attempts.")
        logger.info("Please ensure Ollama is installed and running:")
        logger.info("1. Install Ollama:")
        logger.info("   brew install ollama")
        logger.info("2. Start Ollama server:")
        logger.info("   brew services start ollama")
        logger.info("   # or")
        logger.info("   ollama serve")
        sys.exit(1)

    models_to_setup = models or DEFAULT_MODELS
    current_models = list_models()
    
    logger.info(f"Current models: {current_models}")
    logger.info(f"Models to setup: {models_to_setup}")

    for model in models_to_setup:
        if model not in current_models:
            logger.info(f"Model {model} not found locally. Pulling...")
            if pull_model(model):
                logger.info(f"Successfully pulled model {model}")
            else:
                logger.error(f"Failed to pull model {model}")
        else:
            logger.info(f"Model {model} is already available")

def main():
    parser = argparse.ArgumentParser(description='Setup and manage Ollama models')
    parser.add_argument('--action', choices=['setup', 'list', 'pull', 'delete'],
                       default='setup', help='Action to perform')
    parser.add_argument('--models', nargs='+', help='Specific models to manage')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.action == 'setup':
        setup_models(args.models)
    elif args.action == 'list':
        models = list_models()
        if models:
            logger.info("Available models:")
            for model in models:
                logger.info(f"  - {model}")
        else:
            logger.info("No models found")
    elif args.action == 'pull':
        if not args.models:
            logger.error("Please specify models to pull")
            sys.exit(1)
        for model in args.models:
            if pull_model(model):
                logger.info(f"Successfully pulled model {model}")
            else:
                logger.error(f"Failed to pull model {model}")
    elif args.action == 'delete':
        if not args.models:
            logger.error("Please specify models to delete")
            sys.exit(1)
        for model in args.models:
            if delete_model(model):
                logger.info(f"Successfully deleted model {model}")
            else:
                logger.error(f"Failed to delete model {model}")

if __name__ == "__main__":
    main() 