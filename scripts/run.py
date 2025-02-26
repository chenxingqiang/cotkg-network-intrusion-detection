import os
import sys
import argparse

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from train import train
from model_test import run_test
from config.config import DEFAULT_CONFIG
from config.test_config import TEST_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Run CoTKG-IDS')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], 
                       default='both', help='Run mode')
    parser.add_argument('--test', action='store_true',
                       help='Use test configuration')
    
    args = parser.parse_args()
    config = TEST_CONFIG if args.test else DEFAULT_CONFIG
    
    if args.mode in ['train', 'both']:
        print("Starting training...")
        train(config=config)
    
    if args.mode in ['test', 'both']:
        print("\nStarting testing...")
        run_test(config=config)

if __name__ == "__main__":
    main() 