import unittest
import torch
from src.models.graph_sage_model import GraphSAGE
from src.models.hybrid_model import HybridModel
from torch_geometric.data import Data


class TestModels(unittest.TestCase):
    def setUp(self):
        self.num_features = 10
        self.num_classes = 5
        self.num_nodes = 100
        self.edge_index = torch.randint(0, self.num_nodes, (2, 300))
        self.x = torch.randn(self.num_nodes, self.num_features)
        self.y = torch.randint(0, self.num_classes, (self.num_nodes,))
        self.data = Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def test_graph_sage(self):
        model = GraphSAGE(self.num_features, 64,
                        self.num_classes, num_layers=3, dropout=0.1)
        out = model(self.data.x, self.data.edge_index)
        self.assertEqual(out.size(), (self.num_nodes, self.num_classes))

    def test_hybrid_model(self):
        model = HybridModel(self.num_features, 64,
                            self.num_classes, num_layers=3, dropout=0.1)
        out = model(self.data)
        self.assertEqual(out.size(), (self.num_nodes, self.num_classes))


if __name__ == '__main__':
    unittest.main()
