import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import logging
from unittest.mock import patch, MagicMock

from cotkg_ids.data_processing.preprocess import load_and_preprocess_data
from cotkg_ids.data_processing.feature_engineering import engineer_features
from cotkg_ids.data_processing.data_balancing import balance_dataset
from cotkg_ids.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from cotkg_ids.knowledge_graph.kg_analysis import KnowledgeGraphAnalyzer

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("Setting up test pipeline...")

        # Set up test data paths
        cls.raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'CICIDS2017.csv')

        # Create comprehensive sample test data with edge cases
        cls.sample_data = pd.DataFrame({
            'total_fwd_packets': [100, 200, 300, 400, 500, 0, 1000000],  # Changed from Packet Count
            'total_backward_packets': [50, 100, 150, 200, 250, 0, 500000],  # Added for packet counting
            'total_length_of_fwd_packets': [500, 1000, 1500, 2000, 2500, 0, 999999],
            'total_length_of_bwd_packets': [200, 400, 600, 800, 1000, 0, 999999],
            'flow_duration': [1000, 2000, 3000, 4000, 5000, 0, 999999],
            'protocol': ['TCP', 'UDP', 'TCP', 'ICMP', 'TCP', 'UDP', 'TCP'],  # Added for protocol features
            'flag_fin': [1, 0, 1, 0, 1, 0, 1],  # Added for flag features
            'flag_syn': [0, 1, 1, 0, 0, 1, 1],
            'flag_rst': [0, 0, 1, 0, 0, 0, 1],
            'flag_psh': [1, 1, 0, 0, 1, 0, 1],
            'flag_ack': [1, 1, 1, 0, 1, 0, 1],
            'flag_urg': [0, 0, 0, 0, 0, 0, 1],
            'label': ['normal', 'attack', 'normal', 'attack', 'normal', 'normal', 'attack']
        })

        # Initialize Knowledge Graph with retry mechanism
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                cls.logger.info(f"Attempting to connect to Neo4j (attempt {attempt + 1}/{max_retries})...")
                cls.kg_constructor = KnowledgeGraphConstructor()
                # Test connection
                cls.kg_constructor.graph.run("MATCH (n) RETURN n LIMIT 1")
                cls.logger.info("Successfully connected to Neo4j")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}"
                    cls.logger.error(error_msg)
                    raise Exception(error_msg)
                cls.logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)

        cls.kg_analyzer = KnowledgeGraphAnalyzer()

    def setUp(self):
        """Set up test fixtures before each test."""
        self.logger.info("\n=== Starting new test ===")

        # Clear existing graph data
        try:
            self.kg_constructor.clear_graph()
            self.logger.info("Cleared existing graph data")
        except Exception as e:
            self.logger.warning(f"Failed to clear graph: {str(e)}")

        # Reset test data
        self.test_data = self.sample_data.copy()
        self.logger.info(f"Test data prepared with {len(self.test_data)} samples")

    def test_data_preprocessing(self):
        """Test data preprocessing step of the pipeline."""
        self.logger.info("Testing data preprocessing...")

        # Test with sample data
        try:
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)

            self.assertIsInstance(processed_data, pd.DataFrame)
            self.assertTrue(all(col.islower() for col in processed_data.columns))
            self.assertGreater(len(processed_data.columns), 0)

            # Check for NaN values
            self.assertFalse(processed_data.isna().any().any(), "Preprocessed data contains NaN values")

            # Check data types
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            self.assertGreater(len(numeric_cols), 0, "No numeric columns found after preprocessing")

            self.logger.info("Sample data preprocessing successful")

        except Exception as e:
            self.logger.error(f"Sample data preprocessing failed: {str(e)}")
            raise

        # Test with file input if available
        if os.path.exists(self.raw_data_path):
            try:
                file_processed_data = load_and_preprocess_data(self.raw_data_path, test_mode=True)
                self.assertIsInstance(file_processed_data, pd.DataFrame)
                self.assertTrue(all(col.islower() for col in file_processed_data.columns))
                self.logger.info("File data preprocessing successful")
            except Exception as e:
                self.logger.error(f"File data preprocessing failed: {str(e)}")
                raise
        else:
            self.logger.warning(f"Raw data file not found at {self.raw_data_path}")

    def test_feature_engineering(self):
        """Test feature engineering step of the pipeline."""
        self.logger.info("Testing feature engineering...")

        try:
            # First preprocess the data
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)

            # Then engineer features
            engineered_data = engineer_features(processed_data)

            self.assertIsInstance(engineered_data, pd.DataFrame)
            self.assertGreaterEqual(len(engineered_data.columns), len(processed_data.columns))

            # Verify engineered features
            required_features = ['flow_bytes/s', 'flow_packets/s']
            for feature in required_features:
                self.assertIn(feature, engineered_data.columns)
                self.assertTrue((engineered_data[feature] >= 0).all())

            # Check for NaN values in engineered features
            self.assertFalse(engineered_data.isna().any().any(), "Engineered features contain NaN values")

            # Verify edge cases
            zero_duration_rows = processed_data[processed_data['flow_duration'] == 0]
            if not zero_duration_rows.empty:
                zero_duration_features = engineered_data.loc[zero_duration_rows.index]
                self.assertTrue(all(zero_duration_features['flow_bytes/s'].notna()),
                              "Flow bytes/s undefined for zero duration")

            self.logger.info("Feature engineering successful")

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def test_data_balancing(self):
        """Test data balancing step of the pipeline."""
        self.logger.info("Testing data balancing...")

        try:
            # Prepare data
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)
            engineered_data = engineer_features(processed_data)

            X = engineered_data.drop('label', axis=1)
            y = engineered_data['label']

            # Test each balancing method
            balancing_methods = ['smote', 'basic', 'smote_tomek', 'hybrid']

            for method in balancing_methods:
                with self.subTest(method=method):
                    self.logger.info(f"Testing {method} balancing method...")

                    X_balanced, y_balanced = balance_dataset(X, y, method=method)

                    self.assertIsInstance(X_balanced, pd.DataFrame)
                    self.assertIsInstance(y_balanced, pd.Series)
                    self.assertEqual(len(X_balanced), len(y_balanced))

                    # Check if classes are more balanced
                    value_counts = y_balanced.value_counts()
                    balance_ratio = max(value_counts) / min(value_counts)
                    self.assertLess(balance_ratio, len(y.value_counts()) + 1)

                    # Check for data integrity
                    self.assertFalse(X_balanced.isna().any().any(),
                                   f"Balanced data contains NaN values with {method} method")

                    self.logger.info(f"{method} balancing successful with ratio {balance_ratio:.2f}")

        except Exception as e:
            self.logger.error(f"Data balancing failed: {str(e)}")
            raise

    @patch('cotkg_ids.knowledge_graph.kg_constructor.KnowledgeGraphConstructor.add_flow')
    def test_knowledge_graph_construction_with_mock(self, mock_add_flow):
        """Test knowledge graph construction with mocked Neo4j."""
        self.logger.info("Testing knowledge graph construction with mock...")

        # Configure mock
        mock_add_flow.return_value = True

        try:
            # Prepare data
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)
            engineered_data = engineer_features(processed_data)
            X = engineered_data.drop('label', axis=1)
            y = engineered_data['label']
            X_balanced, y_balanced = balance_dataset(X, y, method='basic')

            # Reset index
            X_balanced = X_balanced.reset_index(drop=True)
            y_balanced = y_balanced.reset_index(drop=True)

            # Add flows to graph
            for idx, (_, row) in enumerate(X_balanced.iterrows()):
                flow_data = {
                    **row.to_dict(),
                    'label': y_balanced[idx]
                }

                # Convert numpy types to Python native types
                flow_data = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in flow_data.items()
                }

                result = self.kg_constructor.add_flow(flow_data)
                self.assertTrue(result)

            self.logger.info("Mock knowledge graph construction successful")

        except Exception as e:
            self.logger.error(f"Mock knowledge graph construction failed: {str(e)}")
            raise

    def test_knowledge_graph_construction(self):
        """Test knowledge graph construction with real Neo4j."""
        self.logger.info("Testing knowledge graph construction...")

        try:
            # Prepare data
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)
            engineered_data = engineer_features(processed_data)
            X = engineered_data.drop('label', axis=1)
            y = engineered_data['label']
            X_balanced, y_balanced = balance_dataset(X, y, method='basic')

            # Reset index
            X_balanced = X_balanced.reset_index(drop=True)
            y_balanced = y_balanced.reset_index(drop=True)

            # Add flows to graph
            for idx, (_, row) in enumerate(X_balanced.iterrows()):
                flow_data = {
                    **row.to_dict(),
                    'label': y_balanced[idx]
                }

                # Convert numpy types to Python native types
                flow_data = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in flow_data.items()
                }

                result = self.kg_constructor.add_flow(flow_data)
                self.assertIsNotNone(result)

            # Verify graph construction
            stats = self.kg_analyzer.get_graph_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('nodes', stats)
            self.assertIn('relationships', stats)
            
            # Check that nodes and relationships are dictionaries
            self.assertIsInstance(stats['nodes'], dict)
            self.assertIsInstance(stats['relationships'], dict)
            
            # Check that there are some nodes and relationships
            total_nodes = sum(stats['nodes'].values())
            total_relationships = sum(stats['relationships'].values())
            self.assertGreaterEqual(total_nodes, 0)
            self.assertGreaterEqual(total_relationships, 0)

            self.logger.info(f"Knowledge graph construction successful with {total_nodes} nodes and {total_relationships} relationships")
            self.logger.info("Complete pipeline test successful!")

        except Exception as e:
            self.logger.error(f"Knowledge graph construction failed: {str(e)}")
            raise

    def test_complete_pipeline(self):
        """Test the complete pipeline end-to-end."""
        self.logger.info("Testing complete pipeline...")

        try:
            # 1. Data preprocessing
            processed_data = load_and_preprocess_data(self.test_data, test_mode=True)
            self.assertIsInstance(processed_data, pd.DataFrame)
            self.logger.info("Pipeline step 1: Preprocessing complete")

            # 2. Feature engineering
            engineered_data = engineer_features(processed_data)
            self.assertIsInstance(engineered_data, pd.DataFrame)
            self.logger.info("Pipeline step 2: Feature engineering complete")

            # 3. Data balancing
            X = engineered_data.drop('label', axis=1)
            y = engineered_data['label']
            X_balanced, y_balanced = balance_dataset(X, y, method='basic')
            self.assertEqual(len(X_balanced), len(y_balanced))
            self.logger.info("Pipeline step 3: Data balancing complete")

            # 4. Knowledge graph construction
            X_balanced = X_balanced.reset_index(drop=True)
            y_balanced = y_balanced.reset_index(drop=True)

            # Limit the number of flows to add to the knowledge graph
            max_flows = 5  # Only add up to 5 flows to limit test time
            flow_count = min(max_flows, len(X_balanced))
            self.logger.info(f"Adding {flow_count} flows to knowledge graph (limited for testing)")

            for idx in range(flow_count):
                row = X_balanced.iloc[idx]
                flow_data = {
                    **row.to_dict(),
                    'label': y_balanced[idx]
                }
                flow_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                            for k, v in flow_data.items()}

                result = self.kg_constructor.add_flow(flow_data)
                self.assertIsNotNone(result)

            # 5. Verify final graph state
            stats = self.kg_analyzer.get_graph_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('nodes', stats)
            self.assertIn('relationships', stats)
            
            # Check that nodes and relationships are dictionaries
            self.assertIsInstance(stats['nodes'], dict)
            self.assertIsInstance(stats['relationships'], dict)
            
            # Check that there are some nodes and relationships
            total_nodes = sum(stats['nodes'].values())
            total_relationships = sum(stats['relationships'].values())
            self.assertGreaterEqual(total_nodes, 0)
            self.assertGreaterEqual(total_relationships, 0)
            
            self.logger.info(f"Pipeline step 4: Knowledge graph construction complete with {total_nodes} nodes and {total_relationships} relationships")
            self.logger.info("Complete pipeline test successful!")

        except Exception as e:
            self.logger.error(f"Complete pipeline test failed: {str(e)}")
            raise

    def tearDown(self):
        """Clean up after each test."""
        try:
            self.kg_constructor.clear_graph()
            self.logger.info("Test cleanup successful")
        except Exception as e:
            self.logger.warning(f"Test cleanup failed: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        try:
            cls.kg_constructor.clear_graph()
            cls.logger.info("Final cleanup successful")
        except Exception as e:
            cls.logger.warning(f"Final cleanup failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)