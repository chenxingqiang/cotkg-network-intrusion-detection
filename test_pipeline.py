import pandas as pd
import numpy as np
from src.data_processing.preprocess import load_and_preprocess_data
from src.data_processing.feature_engineering import engineer_features
from src.data_processing.data_balancing import balance_dataset
from src.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from src.knowledge_graph.kg_analysis import KnowledgeGraphAnalyzer

def test_data_pipeline():
    """Test the complete data processing pipeline"""
    try:
        # 1. Load and preprocess data
        print("\n1. Testing data loading and preprocessing...")
        data_result = load_and_preprocess_data(
            'data/raw/CICIDS2017.csv',
            test_mode=True
        )
        
        if data_result is None:
            raise ValueError("Data preprocessing failed")
            
        X_train, X_test, y_train, y_test = data_result
        print("\nPreprocessing Results:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print("\nFeature names:")
        print(X_train.columns.tolist())
        
        # 2. Test feature engineering
        print("\n2. Testing feature engineering...")
        X_train_engineered = engineer_features(X_train)
        print("\nEngineered Features:")
        new_features = set(X_train_engineered.columns) - set(X_train.columns)
        print("New features added:")
        for feature in new_features:
            print(f"  - {feature}")
        print(f"\nTotal features after engineering: {X_train_engineered.shape[1]}")
        
        # 3. Test data balancing
        print("\n3. Testing data balancing...")
        print("\nTesting different balancing methods:")
        
        for method in ['smote', 'basic', 'smote_tomek', 'hybrid']:
            print(f"\nTesting {method.upper()} balancing:")
            X_balanced, y_balanced = balance_dataset(
                X_train_engineered, 
                y_train,
                method=method
            )
            print(f"Balanced data shape: {X_balanced.shape}")
        
        # 4. Test knowledge graph construction
        print("\n4. Testing knowledge graph construction...")
        kg = KnowledgeGraphConstructor()
        
        # Clear existing data
        kg.clear_graph()
        print("Cleared existing graph data")
        
        # Add all balanced data to graph
        print("\nAdding flows to knowledge graph...")
        
        # Reset index of balanced data to ensure alignment
        X_balanced = X_balanced.reset_index(drop=True)
        y_balanced = y_balanced.reset_index(drop=True)
        
        # Use zip to iterate over X and y together
        for idx, (_, row) in enumerate(X_balanced.iterrows()):
            flow_data = {
                **row.to_dict(),
                'label': y_balanced[idx]  # Use direct indexing instead of iloc
            }
            
            # Convert numpy types to Python native types
            flow_data = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in flow_data.items()
            }
            
            kg.add_flow(flow_data)
            
            if idx % 10 == 0:  # Progress indicator
                print(f"Added {idx} flows...")
        
        # Analyze the graph
        print("\nAnalyzing knowledge graph...")
        analyzer = KnowledgeGraphAnalyzer()
        stats = analyzer.get_graph_statistics()
        print("\nGraph Statistics:")
        print(f"Nodes: {stats['nodes']}")
        print(f"Relationships: {stats['relationships']}")
        
        # Visualize
        print("\nGenerating visualization...")
        analyzer.visualize_graph_sample(limit=20)
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError in test pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting pipeline test...")
    test_data_pipeline() 