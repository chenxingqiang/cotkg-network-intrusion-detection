"""Test script for CoTKG-IDS"""

import os
import torch
import logging
from datetime import datetime
from config.test_config import TEST_CONFIG
from main import run_full_pipeline
from models.graph_sage_model import GraphSAGE, evaluate_graph_sage

def setup_test_logging():
    """Setup logging for tests"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_test(config=None):
    """Run tests with given configuration"""
    setup_test_logging()
    config = config or TEST_CONFIG
    
    logging.info("Starting tests with configuration:")
    logging.info(f"Test config: {config['model']}")
    
    try:
        # Run pipeline with config
        results = run_full_pipeline(config=config)  # Use keyword argument
        
        if results and results['model']:
            # Evaluate model
            model = results['model']
            data = results['graph_data']
            
            accuracy, predictions = evaluate_graph_sage(model, data)
            
            logging.info(f"Test Accuracy: {accuracy:.4f}")
            
            # Save test results
            test_results_path = os.path.join('results', 'test_results.txt')
            with open(test_results_path, 'w') as f:
                f.write(f"Test Accuracy: {accuracy:.4f}\n")
                f.write(f"Number of nodes: {data.num_nodes}\n")
                f.write(f"Number of edges: {data.num_edges}\n")
                f.write(f"Number of features: {data.num_node_features}\n")
            
            return accuracy, predictions
        else:
            logging.error("Testing failed - no results returned")
            return None
            
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    run_test() 