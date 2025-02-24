import os
import pandas as pd
import numpy as np
import torch

from data_processing.preprocess import load_and_preprocess_data
from data_processing.feature_engineering import engineer_features, get_feature_importance
from data_processing.data_balancing import balance_dataset
from knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from knowledge_graph.kg_analysis import KnowledgeGraphAnalyzer
from models.graph_sage_model import GraphSAGE, train_graph_sage

def run_full_pipeline():
    """Run the complete data processing and training pipeline"""
    try:
        # Initialize result dictionary
        results = {
            'model': None,
            'kg': None,
            'stats': None,
            'feature_importance': None
        }
        
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        data_result = load_and_preprocess_data(
            'data/raw/CICIDS2017.csv',
            test_mode=True
        )
        
        if data_result is None:
            raise ValueError("Data preprocessing failed")
            
        X_train, X_test, y_train, y_test = data_result
        
        # 2. Feature engineering
        print("\n2. Engineering features...")
        X_train_engineered = engineer_features(X_train)
        X_test_engineered = engineer_features(X_test)
        
        # Calculate feature importance
        print("\n3. Calculating feature importance...")
        importance = get_feature_importance(
            pd.concat([X_train_engineered, pd.Series(y_train, name='label')], axis=1)
        )
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        # 4. Data balancing
        print("\n4. Balancing data...")
        X_balanced, y_balanced = balance_dataset(
            X_train_engineered, 
            y_train,
            method='hybrid'
        )
        
        # 5. Knowledge graph construction
        print("\n5. Constructing knowledge graph...")
        kg = KnowledgeGraphConstructor()
        kg.clear_graph()
        
        # Add balanced data to graph
        print("\nAdding flows to knowledge graph...")
        for idx, (_, row) in enumerate(X_balanced.iterrows()):
            flow_data = {
                **row.to_dict(),
                'label': y_balanced[idx]
            }
            kg.add_flow(flow_data)
            
            if idx % 50 == 0:
                print(f"Added {idx} flows...")
        
        # 6. Analyze knowledge graph
        print("\n6. Analyzing knowledge graph...")
        analyzer = KnowledgeGraphAnalyzer()
        
        # Get basic statistics
        stats = analyzer.get_graph_statistics()
        print("\nKnowledge Graph Statistics:")
        print("Nodes:", stats['nodes'])
        print("Relationships:", stats['relationships'])
        
        # Get attack distribution
        attack_dist = analyzer.get_attack_distribution()
        print("\nAttack Distribution:")
        print(attack_dist)
        
        # Get feature statistics
        feature_stats = analyzer.get_feature_statistics()
        print("\nFeature Statistics:")
        print(feature_stats)
        
        # Generate visualization
        print("\nGenerating graph visualization...")
        analyzer.visualize_graph_sample(limit=50)
        
        # 7. Train GraphSAGE model
        print("\n7. Training GraphSAGE model...")
        graph_data = kg.prepare_data_for_graph_sage(X_balanced, y_balanced)
        
        if graph_data is not None:
            model = GraphSAGE(
                in_channels=X_balanced.shape[1],
                hidden_channels=64,
                out_channels=len(y_balanced.unique()),
                num_layers=2,
                dropout=0.5
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            train_graph_sage(model, graph_data, optimizer)
            
            # Update results dictionary
            results['model'] = model
        else:
            print("Warning: Could not prepare graph data for training")
        
        # 8. Save results
        print("\n8. Saving results...")
        os.makedirs('results', exist_ok=True)
        
        # Save feature importance
        importance.to_csv('results/feature_importance.csv')
        
        # Save model statistics
        with open('results/model_stats.txt', 'w') as f:
            f.write("Knowledge Graph Statistics:\n")
            f.write(f"Nodes: {stats['nodes']}\n")
            f.write(f"Relationships: {stats['relationships']}\n")
            
            f.write("\nAttack Distribution:\n")
            f.write(str(attack_dist))
            
            f.write("\nFeature Statistics:\n")
            f.write(str(feature_stats))
        
        print("\nPipeline completed successfully!")
        print("Results have been saved to the 'results' directory")
        
        # Update remaining results
        results['kg'] = kg
        results['stats'] = stats
        results['feature_importance'] = importance
        
        return results
        
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_full_pipeline()
