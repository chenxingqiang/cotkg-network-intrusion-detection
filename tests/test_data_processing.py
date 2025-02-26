# tests/test_data_processing.py

from cotkg_ids.knowledge_graph.cot_generator import generate_cot, parse_cot_response
from cotkg_ids.data_processing.feature_engineering import engineer_features
from cotkg_ids.data_processing.preprocess import load_and_preprocess_data
import unittest
import pandas as pd
import os
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 将项目根目录添加到 Python 路径
sys.path.append(str(PROJECT_ROOT))


class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # 设置测试数据文件路径
        cls.raw_data_path = os.path.join(
            PROJECT_ROOT, 'data', 'raw', 'CICIDS2017.csv')

        # 创建示例数据
        cls.sample_data = pd.DataFrame({
            'Packet Count': [100, 200, 300],
            'Bytes Transferred': [1000, 2000, 3000],
            'Duration': ['1.0s', '2.0s', '3.0s'],
            'Total Length of Fwd Packets': [500, 1000, 1500],
            'Total Length of Bwd Packets': [200, 400, 600],
            'Flow Duration': [1000, 2000, 3000],
            'Flow Bytes/s': [1000, 1000, 1000],
            'Flow Packets/s': [100, 100, 100]
        })

    def test_load_and_preprocess_data(self):
        """Test data loading and preprocessing."""
        try:
            # Test using DataFrame input
            processed_data = load_and_preprocess_data(self.sample_data.copy())
            self.assertIsInstance(processed_data, pd.DataFrame)
            self.assertIn('total_length_of_fwd_packets', processed_data.columns)

            # Test file loading if raw data file exists
            if os.path.exists(self.raw_data_path):
                file_processed_data = load_and_preprocess_data(
                    self.raw_data_path)
                self.assertIsInstance(file_processed_data, pd.DataFrame)

        except Exception as e:
            self.fail(
                f"load_and_preprocess_data raised an exception: {str(e)}")

    def test_engineer_features(self):
        """Test feature engineering."""
        try:
            engineered_data = engineer_features(self.sample_data)

            # Check if required features are created
            required_features = ['Flow Bytes/s', 'Flow Packets/s']
            for feature in required_features:
                self.assertIn(feature, engineered_data.columns)

            # Check calculations are valid
            self.assertTrue((engineered_data['Flow Bytes/s'] >= 0).all())
            self.assertTrue((engineered_data['Flow Packets/s'] >= 0).all())

        except Exception as e:
            self.fail(f"engineer_features raised an exception: {str(e)}")


class TestCoT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Packet Count': [10, 20, 30],
            'Bytes Transferred': [1000, 2000, 3000],
            'Duration': [5, 10, 15],
            'Total Length of Fwd Packets': [500, 1000, 1500],
            'Total Length of Bwd Packets': [500, 1000, 1500],
            'Flow Duration': [5000, 10000, 15000],
            'Flow Bytes/s': [200, 200, 200],
            'Flow Packets/s': [2, 2, 2]
        })
        self.raw_data_path = 'data/raw/test_data.csv'

    def test_load_and_preprocess_data(self):
        """Test data loading and preprocessing."""
        try:
            # Test using DataFrame input
            sample_data = pd.DataFrame({
                'Packet Count': [10, 20, 30],
                'Bytes Transferred': [1000, 2000, 3000],
                'Duration': [5, 10, 15],
                'Total Length of Fwd Packets': [500, 1000, 1500],
                'Total Length of Bwd Packets': [500, 1000, 1500]
            })
            processed_data = load_and_preprocess_data(sample_data)
            self.assertIsInstance(processed_data, pd.DataFrame)
            self.assertIn('total_length_of_fwd_packets', processed_data.columns)

            # Test file loading if raw data file exists
            if os.path.exists(self.raw_data_path):
                file_processed_data = load_and_preprocess_data(
                    self.raw_data_path)
                self.assertIsInstance(file_processed_data, pd.DataFrame)

        except Exception as e:
            self.fail(
                f"load_and_preprocess_data raised an exception: {str(e)}")

    def test_parse_cot_response(self):
        """Test parsing of CoT response."""
        try:
            # Create a more realistic sample response
            self.sample_response = """
            Based on the analysis, I've identified the following:
            
            Entities:
            1. Attack Type: DDoS Attack
               Severity: High
               
            2. Feature: Flow Duration
               Value: Abnormally high
               Importance: Critical
               
            3. Feature: Packet Count
               Value: Excessive
               Importance: High
            
            Relationships:
            1. Flow Duration INDICATES DDoS Attack (confidence: 0.9)
            2. Packet Count INDICATES DDoS Attack (confidence: 0.85)
            """
            
            entities, relationships = parse_cot_response(self.sample_response)
            
            # Verify entity extraction
            self.assertGreater(len(entities), 0, "No entities were extracted")
            self.assertTrue(
                any(e['type'] == 'Attack' for e in entities),
                "No attack type was identified"
            )
            self.assertTrue(
                any(e['type'] == 'Feature' for e in entities),
                "No features were extracted"
            )
            
            # Verify relationship extraction
            self.assertGreater(len(relationships), 0,
                               "No relationships were extracted")
            
            # Print extracted content for debugging
            print("\nExtracted Entities:", entities)
            print("\nExtracted Relationships:", relationships)
            
            # Check relationship completeness
            for rel in relationships:
                self.assertIn('source', rel)
                self.assertIn('type', rel)
                self.assertIn('target', rel)
            
        except Exception as e:
            self.fail(f"parse_cot_response raised an exception: {str(e)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
