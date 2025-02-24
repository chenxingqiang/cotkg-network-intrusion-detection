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
from config.config import DEFAULT_CONFIG

def run_full_pipeline(config=None):
    """
    Run the complete data processing and training pipeline
    
    Args:
        config (dict, optional): Configuration dictionary. Defaults to DEFAULT_CONFIG.
    """
    try:
        # Use default config if none provided
        config = config or DEFAULT_CONFIG
        
        # Initialize result dictionary
        results = {
            'model': None,
            'kg': None,
            'stats': None,
            'feature_importance': None,
            'graph_data': None,
            'cot_analysis': []  # Add COT analysis results
        }
        
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        data_result = load_and_preprocess_data(
            config['data']['raw_data_path'],
            test_mode=True,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
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
        results['feature_importance'] = importance
        
        # 4. Data balancing
        print("\n4. Balancing data...")
        X_balanced, y_balanced = balance_dataset(
            X_train_engineered, 
            y_train,
            method=config['data']['balancing']['method']
        )
        
        # 5. Knowledge graph construction with COT
        print("\n5. Constructing knowledge graph with Chain of Thought analysis...")
        kg = KnowledgeGraphConstructor(
            uri=config['neo4j']['uri'],
            username=config['neo4j']['username'],
            password=config['neo4j']['password'],
            max_retries=config['neo4j']['max_retries']
        )
        kg.clear_graph()
        
        # Add balanced data to graph with COT analysis
        print("\nAdding flows to knowledge graph with COT analysis...")
        for idx, (_, row) in enumerate(X_balanced.iterrows()):
            flow_data = {
                **row.to_dict(),
                'label': y_balanced[idx]
            }
            kg.add_flow(flow_data)
            
            if idx % 50 == 0:
                print(f"Added {idx} flows...")
        
        results['kg'] = kg
        
        # 6. Analyze knowledge graph
        print("\n6. Analyzing knowledge graph...")
        analyzer = KnowledgeGraphAnalyzer()
        stats = analyzer.get_graph_statistics()
        results['stats'] = stats
        
        # 7. Train GraphSAGE model
        print("\n7. Training GraphSAGE model...")
        graph_data = kg.prepare_data_for_graph_sage(X_balanced, y_balanced)
        results['graph_data'] = graph_data
        
        if graph_data is not None:
            model = GraphSAGE(
                in_channels=X_balanced.shape[1],
                hidden_channels=config['model']['graphsage']['hidden_channels'],
                out_channels=len(y_balanced.unique()),
                num_layers=config['model']['graphsage']['num_layers'],
                dropout=config['model']['graphsage']['dropout']
            )
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['model']['graphsage']['learning_rate'],
                weight_decay=config['model']['graphsage']['weight_decay']
            )
            
            train_graph_sage(
                model, 
                graph_data, 
                optimizer,
                epochs=config['model']['training']['epochs']
            )
            
            results['model'] = model
        
        # 8. Save results
        print("\n8. Saving results...")
        
        # Save feature importance
        importance.to_csv('results/feature_importance.csv')
        
        # Save model statistics
        with open('results/model_stats.txt', 'w') as f:
            f.write("Knowledge Graph Statistics:\n")
            f.write(f"Nodes: {stats['nodes']}\n")
            f.write(f"Relationships: {stats['relationships']}\n")
            
            f.write("\nFeature Statistics:\n")
            f.write(str(analyzer.get_feature_statistics()))
        
        print("\nPipeline completed successfully!")
        print("Results have been saved to the 'results' directory")
        
        return results
        
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_full_pipeline()
