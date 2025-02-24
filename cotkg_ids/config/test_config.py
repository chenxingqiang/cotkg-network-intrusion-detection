"""Test configuration for CoTKG-IDS"""

from .config import DEFAULT_CONFIG

# Test configuration with smaller dataset and faster training
TEST_CONFIG = DEFAULT_CONFIG.copy()

# Update data config for testing
TEST_CONFIG['data'].update({
    'test_size': 0.3,
    'sample_size': 1000  # Use smaller dataset for testing
})

# Update model config for testing
TEST_CONFIG['model']['graphsage'].update({
    'hidden_channels': 32,
    'num_layers': 2,
    'dropout': 0.3
})

TEST_CONFIG['model']['training'].update({
    'epochs': 20,
    'batch_size': 16,
    'early_stopping': {
        'patience': 5,
        'min_delta': 0.01
    }
}) 