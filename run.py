import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from main import run_full_pipeline

if __name__ == "__main__":
    results = run_full_pipeline() 