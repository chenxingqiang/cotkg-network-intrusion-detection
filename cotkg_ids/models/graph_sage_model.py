import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        """
        Initialize GraphSAGE model with optimized architecture
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList()
        
        # Create list of convolution layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Additional components
        self.residual = num_layers > 1
        self.act = torch.nn.PReLU()  # Learnable ReLU
        
    def forward(self, x, edge_index):
        """Forward pass with skip connections and batch normalization"""
        # Initial features
        h = x
        
        for i in range(self.num_layers - 1):
            # Main path
            x_main = self.convs[i](x, edge_index)
            x_main = self.batch_norms[i](x_main)
            x_main = self.act(x_main)
            x_main = F.dropout(x_main, p=self.dropout, training=self.training)
            
            # Residual connection if dimensions match
            if self.residual and x_main.shape[-1] == x.shape[-1]:
                x = x_main + x
            else:
                x = x_main
        
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
    Train GraphSAGE model with improved training process
    """
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Calculate loss with class weights
        class_weights = calculate_class_weights(data.y[data.train_mask])
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        # L2 regularization
        l2_lambda = 1e-4
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        loss = loss + l2_lambda * l2_reg
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.nll_loss(val_out[data.val_mask], data.y[data.val_mask])
            
            # Print metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                train_acc = calculate_accuracy(out[data.train_mask], data.y[data.train_mask])
                val_acc = calculate_accuracy(val_out[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
        
        model.train()


def calculate_accuracy(output, labels):
    """Calculate accuracy for given output and labels"""
    pred = output.argmax(dim=1)
    correct = pred == labels
    return int(correct.sum()) / int(correct.shape[0])


def calculate_class_weights(labels):
    """Calculate class weights to handle imbalanced data"""
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights


def evaluate_graph_sage(model, data):
    """
    Evaluate GraphSAGE model with detailed metrics
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Calculate metrics for test set
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        
        # Calculate per-class accuracy
        classes = torch.unique(data.y)
        per_class_acc = {}
        
        for c in classes:
            mask = (data.y[data.test_mask] == c)
            if mask.sum() > 0:
                class_acc = int((pred[data.test_mask][mask] == c).sum()) / int(mask.sum())
                per_class_acc[c.item()] = class_acc
        
        print("\nPer-class Test Accuracy:")
        for class_idx, acc in per_class_acc.items():
            print(f"Class {class_idx}: {acc:.4f}")
        
        print(f"\nOverall Test Accuracy: {test_acc:.4f}")
    
    return test_acc, pred
