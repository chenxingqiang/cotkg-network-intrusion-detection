# tests/test_data_processing.py

from src.knowledge_graph.cot_generator import generate_cot, parse_cot_response
from src.data_processing.feature_engineering import engineer_features
from src.data_processing.preprocess import load_and_preprocess_data
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
            # 测试使用DataFrame输入
            processed_data = load_and_preprocess_data(self.sample_data)
            self.assertIsInstance(processed_data, pd.DataFrame)
            self.assertIn('Total Length of Fwd Packets',
                          processed_data.columns)

            # 如果原始数据文件存在，也测试文件加载
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

            # 检查是否创建了新特征
            required_features = ['BytesPerPacket',
                                 'PacketsPerSecond', 'BytesPerSecond']
            for feature in required_features:
                self.assertIn(feature, engineered_data.columns)

            # 检查计算是否正确
            self.assertTrue((engineered_data['BytesPerPacket'] >= 0).all())
            self.assertTrue((engineered_data['PacketsPerSecond'] >= 0).all())
            self.assertTrue((engineered_data['BytesPerSecond'] >= 0).all())

        except Exception as e:
            self.fail(f"engineer_features raised an exception: {str(e)}")


class TestCoT(unittest.TestCase):
    def setUp(self):
        """Set up test data for CoT testing."""
        self.sample_flow = """
        Source IP: 192.168.1.100
        Destination IP: 10.0.0.5
        Protocol: TCP
        Source Port: 45123
        Destination Port: 80
        Packet Count: 1000
        Bytes Transferred: 150000
        Duration: 5.2s
        """

        self.sample_response = """
        ### 1. Identify the key features that stand out in this flow:
        - Source IP: 192.168.1.100 (Internal network)
        - Destination Port: 80 (HTTP)
        - High packet count: 1000 packets
        - Large data transfer: 150000 bytes
        
        ### 2. Compare these features to known patterns:
        The combination of high packet count and HTTP port suggests potential web-based activity.
        
        ### 3. Consider any anomalies:
        The rapid packet transmission rate is unusual for normal web traffic.
        
        ### 4. Hypothesis:
        Based on the analysis, this flow most likely represents a Port Scan attack.
        The high packet rate and HTTP targeting are typical indicators.
        
        ### 5. Reasoning:
        - High packet count in short duration
        - Targeting web service port
        - Internal source IP
        
        ### 6. Additional context needed:
        - Source port pattern
        - Payload analysis
        - Similar flows from same source
        """

    def test_generate_cot(self):
        """Test Chain of Thought generation."""
        try:
            cot_response = generate_cot(self.sample_flow)
            self.assertIsInstance(cot_response, str)
            self.assertGreater(len(cot_response), 0)
        except Exception as e:
            self.fail(f"generate_cot raised an exception: {str(e)}")

    def test_parse_cot_response(self):
        """Test parsing of CoT response."""
        try:
            entities, relationships = parse_cot_response(self.sample_response)

            # 验证实体提取
            self.assertGreater(len(entities), 0, "No entities were extracted")
            self.assertTrue(
                any(e['type'] == 'Attack' for e in entities),
                "No attack type was identified"
            )
            self.assertTrue(
                any(e['type'] == 'Feature' for e in entities),
                "No features were extracted"
            )

            # 验证关系提取
            self.assertGreater(len(relationships), 0,
                               "No relationships were extracted")

            # 打印提取的内容用于调试
            print("\nExtracted Entities:", entities)
            print("\nExtracted Relationships:", relationships)

            # 检查关系的完整性
            for rel in relationships:
                self.assertIn('source', rel)
                self.assertIn('type', rel)
                self.assertIn('target', rel)

        except Exception as e:
            self.fail(f"parse_cot_response raised an exception: {str(e)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
