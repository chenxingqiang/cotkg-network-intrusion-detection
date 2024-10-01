from src.knowledge_graph.cot_generator import generate_cot, parse_cot_response
import unittest
import pandas as pd
from src.data_processing.preprocess import load_and_preprocess_data
from src.data_processing.feature_engineering import engineer_features


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Timestamp': ['2017-07-05 10:10:10', '2017-07-05 10:10:11'],
            'Flow Duration': [1000, 2000],
            'Total Fwd Packets': [10, 20],
            'Total Backward Packets': [5, 10],
            'Label': ['BENIGN', 'DDoS']
        })

    def test_load_and_preprocess_data(self):
        processed_data = load_and_preprocess_data(self.sample_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data.columns),
                          len(self.sample_data.columns))

    def test_engineer_features(self):
        engineered_data = engineer_features(self.sample_data)
        self.assertIsInstance(engineered_data, pd.DataFrame)
        self.assertGreater(len(engineered_data.columns),
                          len(self.sample_data.columns))
        self.assertIn('Hour', engineered_data.columns)
        self.assertIn('DayOfWeek', engineered_data.columns)


class TestCoT(unittest.TestCase):
    def test_generate_cot(self):
        sample_flow = {
            'Src IP': '192.168.1.1',
            'Dst IP': '10.0.0.1',
            'Protocol': 'TCP',
            'Timestamp': '2023-06-01 12:00:00',
            'Flow Duration': 1000,
            'Total Fwd Packets': 10,
            'Total Backward Packets': 5
        }
        cot_response = generate_cot(sample_flow)
        self.assertIsInstance(cot_response, str)
        self.assertGreater(len(cot_response), 0)

    def test_parse_cot_response(self):
        sample_response = """
        The flow represents a TCP connection between 192.168.1.1 and 10.0.0.1.
        Key features:
        1. Duration: 1000 ms
        2. Forward packets: 10
        3. Backward packets: 5
        This pattern suggests normal traffic, but further analysis is needed.
        """
        entities, relationships = parse_cot_response(sample_response)
        self.assertIsInstance(entities, list)
        self.assertIsInstance(relationships, list)
        self.assertGreater(len(entities), 0)
        self.assertGreater(len(relationships), 0)

if __name__ == '__main__':
    unittest.main()
