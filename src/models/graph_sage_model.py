import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        """
        Initialize GraphSAGE model
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output classes
            num_layers (int): Number of GraphSAGE layers
            dropout (float): Dropout probability
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create list of convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        """Forward pass"""
        # Initial features
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


def prepare_graph_data(features, edge_index, labels=None):
    """
    Prepare data for GraphSAGE model
    
    Args:
        features (torch.Tensor): Node features
        edge_index (torch.Tensor): Edge indices
        labels (torch.Tensor, optional): Node labels
    
    Returns:
        Data: PyTorch Geometric Data object
    """
    data = Data(
        x=features,
        edge_index=edge_index
    )
    
    if labels is not None:
        data.y = labels
    
    return data


def train_graph_sage(model, data, optimizer=None, epochs=100):
    """
    Train GraphSAGE model
    
    Args:
        model (GraphSAGE): The model to train
        data (Data): PyTorch Geometric Data object
        optimizer (torch.optim.Optimizer, optional): Optimizer to use
        epochs (int): Number of epochs to train
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')


def evaluate_graph_sage(model, data):
    """
    Evaluate GraphSAGE model
    
    Args:
        model (GraphSAGE): The model to evaluate
        data (Data): PyTorch Geometric Data object
    
    Returns:
        float: Accuracy score
        torch.Tensor: Predicted labels
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
    
    return acc, pred
