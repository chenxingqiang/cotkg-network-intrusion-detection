"""Training script for CoTKG-IDS"""

import os
import torch
import logging
from datetime import datetime
from config.config import DEFAULT_CONFIG
from src.main import run_full_pipeline

def setup_logging():
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train(config=None):
    """Train the model with given configuration"""
    setup_logging()
    config = config or DEFAULT_CONFIG
    
    logging.info("Starting training with configuration:")
    logging.info(f"Model config: {config['model']}")
    
    try:
        # Run pipeline with config
        results = run_full_pipeline(config=config)  # Use keyword argument
        
        if results and results['model']:
            # Save model
            model_path = os.path.join('results', 'models', 'latest_model.pt')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(results['model'].state_dict(), model_path)
            
            logging.info(f"Model saved to {model_path}")
            return results
        else:
            logging.error("Training failed - no results returned")
            return None
            
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    train()