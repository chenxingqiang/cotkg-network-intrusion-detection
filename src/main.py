import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from py2neo import Graph, DatabaseError, ServiceUnavailable

from src.data_processing.preprocess import load_and_preprocess_data, validate_required_columns
from src.data_processing.feature_engineering import engineer_features
from src.data_processing.data_balancing import balance_dataset
from src.data_processing.feature_selection import select_features
from src.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from src.models.graph_sage_model import GraphSAGE, train_graph_sage, evaluate_graph_sage
from src.explainability.integrated_gradients import ExplainabilityAnalyzer
from src.visualization.kg_visualizer import visualize_knowledge_graph, visualize_feature_importance
from src.evaluation.metrics import evaluate_model


def main():
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data_result = load_and_preprocess_data('data/raw/CICIDS2017.csv', test_mode=True)

        if data_result is None:
            raise ValueError("Data preprocessing failed")

        X_train, X_test, y_train, y_test = data_result

        # Engineer features
        print("Engineering features...")
        X_train_engineered = engineer_features(X_train)
        X_test_engineered = engineer_features(X_test)

        if X_train_engineered is None or X_test_engineered is None:
            raise ValueError("Feature engineering failed")

        # Balance dataset
        print("Balancing dataset...")
        X_train_balanced, y_train_balanced = balance_dataset(X_train_engineered, y_train)

        if X_train_balanced is None or y_train_balanced is None:
            raise ValueError("Dataset balancing failed")

        # Select features
        print("Selecting features...")
        X_train_selected, selected_features = select_features(X_train_balanced, y_train_balanced)
        X_test_selected = X_test_engineered[selected_features]

        # Initialize knowledge graph
        print("Initializing knowledge graph...")
        try:
            kg_constructor = KnowledgeGraphConstructor(
                'bolt://localhost:7687', 'neo4j', 'neo4jneo4j')
        except (ConnectionRefusedError, ServiceUnavailable) as e:
            print(f"Error connecting to the Neo4j database: {str(e)}")
            return

        # Process each flow and update knowledge graph
        print("Updating knowledge graph...")
        for idx, flow in X_train_balanced.iterrows():
            # Convert flow to dictionary
            flow_dict = flow.to_dict()

            # Add label information
            flow_dict['label'] = y_train_balanced.iloc[idx]

            # Validate required columns
            required_columns = {'flow_duration', 'Protocol Type', 'label'}
            missing_columns = required_columns - set(flow_dict.keys())

            if missing_columns:
                print(f"Warning: Missing required columns for row {idx}: {missing_columns}")
                continue

            # Handle NaN values
            flow_dict = {k: (0.0 if pd.isna(v) and isinstance(v, (float, int)) else v)
                        for k, v in flow_dict.items()}

            # Add to knowledge graph
            try:
                kg_constructor.add_flow(flow_dict)
            except Exception as e:
                print(f"Error adding flow to knowledge graph at index {idx}: {str(e)}")
                continue

        # Prepare data for GraphSAGE
        print("Preparing data for GraphSAGE...")
        graph_data = kg_constructor.prepare_data_for_graph_sage(X_train_balanced, y_train_balanced)

        if graph_data is None:
            print("Error: Data preparation for GraphSAGE failed.")
            return

        # Initialize and train GraphSAGE model
        print("Training GraphSAGE model...")
        model = GraphSAGE(
            in_channels=graph_data.num_features,
            hidden_channels=64,
            out_channels=len(set(y_train_balanced)),
            num_layers=3,
            dropout=0.5
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        try:
            train_graph_sage(model, graph_data, optimizer)
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return

        # Evaluate model
        print("Evaluating model...")
        try:
            acc, y_pred = evaluate_graph_sage(model, graph_data)
            cm, report = evaluate_model(y_test, y_pred, class_names=sorted(set(y_test)))
            print(f"Accuracy: {acc:.4f}")
            print("\nClassification Report:")
            print(report)
        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            return

        # Generate explanations
        print("Generating explanations...")
        try:
            explainer = ExplainabilityAnalyzer(model)
            for i in range(min(5, len(y_test))):  # Explain first 5 test samples
                ig_attributions = explainer.integrated_gradients_explanation(
                    graph_data, target_class=y_test.iloc[i])
                shap_values = explainer.shap_explanation(
                    graph_data, X_train_balanced.iloc[:100])
                feature_importance = explainer.interpret_attributions(
                    ig_attributions, selected_features)
                visualize_feature_importance(feature_importance)
        except Exception as e:
            print(f"Error generating explanations: {str(e)}")

        # Visualize knowledge graph
        print("Visualizing knowledge graph...")
        try:
            visualize_knowledge_graph(kg_constructor.nx_graph)
        except Exception as e:
            print(f"Error visualizing knowledge graph: {str(e)}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return

if __name__ == '__main__':
    main()
