DEFAULT_CONFIG = {
    'data_path': 'data/raw/CICIDS2017.csv',
    'required_columns': [
        'flow_duration',  # 更新列名
        'Header_Length',
        'Protocol Type',
        'Rate',
        'Srate',
        'Drate',
        'label'
    ],
    'test_size': 0.2,
    'random_state': 42,
    'hidden_dim': 64,
    'output_dim': 2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
} 