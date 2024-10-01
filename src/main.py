import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data_processing.preprocess import load_and_preprocess_data
from src.data_processing.feature_engineering import engineer_features
from src.data_processing.data_balancing import balance_dataset
from src.data_processing.feature_selection import select_features
from src.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from src.models.graph_sage_model import GraphSAGE, train_graph_sage, evaluate_graph_sage
from src.explainability.integrated_gradients import ExplainabilityAnalyzer
from src.visualization.kg_visualizer import visualize_knowledge_graph, visualize_feature_importance
from src.evaluation.metrics import evaluate_model


def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('data/raw/CICIDS2017.csv')

    # Engineer features
    print("Engineering features...")
    df = engineer_features(df)

    # Split features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Balance dataset
    print("Balancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y)

    # Select features
    print("Selecting features...")
    X_selected, selected_features = select_features(X_balanced, y_balanced)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.2, random_state=42)

    # Initialize knowledge graph
    print("Initializing knowledge graph...")
    kg_constructor = KnowledgeGraphConstructor(
        'bolt://localhost:7687', 'neo4j', 'password')

    # Process each flow and update knowledge graph
    print("Updating knowledge graph...")
    for _, flow in X_train.iterrows():
        kg_constructor.add_flow(flow.to_dict())

    # Prepare data for GraphSAGE
    print("Preparing data for GraphSAGE...")
    data = kg_constructor.prepare_data_for_graph_sage(X_train, y_train)

    # Initialize and train GraphSAGE model
    print("Training GraphSAGE model...")
    model = GraphSAGE(in_channels=data.num_features, hidden_channels=64,
                      out_channels=len(y.unique()), num_layers=3, dropout=0.5)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    train_graph_sage(model, data, optimizer)

    # Evaluate model
    print("Evaluating model...")
    acc, y_pred = evaluate_graph_sage(model, data)
    cm, report = evaluate_model(y_test, y_pred, class_names=y.unique())

    # Generate explanations
    print("Generating explanations...")
    explainer = ExplainabilityAnalyzer(model)
    for i in range(5):  # Explain first 5 test samples
        ig_attributions = explainer.integrated_gradients_explanation(
            data, target_class=y_test.iloc[i])
        shap_values = explainer.shap_explanation(
            data, X_train.iloc[:100])  # Use first 100 samples as background
        feature_importance = explainer.interpret_attributions(
            ig_attributions, selected_features)
        visualize_feature_importance(feature_importance)

    # Visualize knowledge graph
    print("Visualizing knowledge graph...")
    visualize_knowledge_graph(kg_constructor.nx_graph)


if __name__ == '__main__':
    main()
