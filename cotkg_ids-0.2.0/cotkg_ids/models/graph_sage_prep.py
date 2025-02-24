import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class GraphSagePrep:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_edges(self, num_nodes):
        """
        Create edges for the graph based on number of nodes.
        Returns edge indices in COO format.
        """
        # Create edges connecting each node to its neighbors
        edges = []
        for i in range(num_nodes):
            # Connect to previous node if exists
            if i > 0:
                edges.append([i-1, i])
                edges.append([i, i-1])  # bidirectional

            # Connect to next node if exists
            if i < num_nodes - 1:
                edges.append([i, i+1])
                edges.append([i+1, i])  # bidirectional

        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def prepare_data(self, features, labels):
        """
        Prepare data for GraphSAGE model.

        Args:
            features: numpy array of feature vectors
            labels: numpy array of labels

        Returns:
            PyTorch Geometric Data object
        """
        # Scale features
        X = self.scaler.fit_transform(features)
        X = torch.FloatTensor(X)

        # Convert labels to tensor
        y = torch.LongTensor(labels)

        # Create edges
        edge_index = self.create_edges(len(features))

        # Create PyG Data object
        data = Data(x=X, edge_index=edge_index, y=y)

        return data

    def prepare_batch(self, features):
        """
        Prepare a single batch of data for inference.

        Args:
            features: numpy array of feature vectors

        Returns:
            PyTorch Geometric Data object
        """
        # Scale features using fitted scaler
        X = self.scaler.transform(features)
        X = torch.FloatTensor(X)

        # Create edges
        edge_index = self.create_edges(len(features))

        # Create PyG Data object
        data = Data(x=X, edge_index=edge_index)

        return data