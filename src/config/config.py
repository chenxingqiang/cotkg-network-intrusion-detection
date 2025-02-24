"""Configuration settings for CoTKG-IDS"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Neo4j configuration
NEO4J_CONFIG = {
    'uri': "bolt://localhost:7687",
    'username': "neo4j",
    'password': "neo4jneo4j",
    'max_retries': 3
}

# Data processing configuration
DATA_CONFIG = {
    'raw_data_path': os.path.join(DATA_DIR, 'raw', 'CICIDS2017.csv'),
    'test_size': 0.2,
    'random_state': 42,
    'feature_selection': {
        'k_best': 150,
        'method': 'mutual_info_classif'
    },
    'balancing': {
        'method': 'smote_tomek',
        'sampling_strategy': 'auto',
        'random_state': 42
    }
}

# Knowledge Graph configuration
KG_CONFIG = {
    'similarity_threshold': 0.7,
    'edge_weight_threshold': 0.4,
    'max_edges_per_node': 100,
    'feature_importance_threshold': 0.005
}

# Model configuration
MODEL_CONFIG = {
    'graphsage': {
        'hidden_channels': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    },
    'training': {
        'epochs': 200,
        'batch_size': 64,
        'early_stopping': {
            'patience': 15,
            'min_delta': 0.0005
        },
        'validation_split': 0.2
    }
}

# Visualization configuration
VIZ_CONFIG = {
    'max_nodes_display': 100,
    'node_size': 50,
    'edge_width': 0.5,
    'font_size': 8
}

# COT configuration
COT_CONFIG = {
    'provider': ['ollama', 'qianwen'],
    'model': 'llama2',
    'max_tokens': 1500,
    'temperature': 0.85,
    'top_p': 0.8,
    'ollama': {
        'base_url': 'http://localhost:11434',
        'timeout': 30,
        'models': ['llama2', 'mistral', 'codellama', 'vicuna'],
        'context_window': 4096,
        'stream': False
    },
    'qianwen': {
        'model': 'qwen-max',
        'max_tokens': 1500,
        'temperature': 0.85,
        'top_p': 0.8,
        'context_window': 8192,
        'stream': False
    }
}

# Default configuration
DEFAULT_CONFIG = {
    'neo4j': NEO4J_CONFIG,
    'data': DATA_CONFIG,
    'kg': KG_CONFIG,
    'model': MODEL_CONFIG,
    'viz': VIZ_CONFIG,
    'cot': COT_CONFIG
} 