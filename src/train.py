import pandas as pd
from src.data_processing.preprocess import load_and_preprocess_data
from src.models.graph_sage_prep import prepare_graph_sage_data
from src.models.graph_sage import GraphSAGE

def train_model(config):
    """Main training function"""
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            config['data_path'],
            test_mode=True,
            test_size=config['test_size'],
            random_state=config['random_state']
        )

        # Validate data
        if X_train is None or y_train is None:
            raise ValueError("Data preprocessing failed")

        # Prepare GraphSAGE data
        graph_data = prepare_graph_sage_data(pd.concat([X_train, y_train], axis=1))
        if graph_data is None:
            raise ValueError("GraphSAGE data preparation failed")

        node_features, edges, labels = graph_data

        # Initialize and train model
        model = GraphSAGE(
            input_dim=node_features.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        )

        # Training loop...

    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None