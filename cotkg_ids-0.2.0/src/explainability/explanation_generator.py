import networkx as nx
from torch_geometric.nn import GNNExplainer


class ExplanationGenerator:
    def __init__(self, model, kg_constructor):
        self.model = model
        self.kg_constructor = kg_constructor
        self.gnn_explainer = GNNExplainer(model, num_hops=3)

    def generate_explanation(self, node_idx, data):
        # Get GNN explanation
        node_feat_mask, edge_mask = self.gnn_explainer.explain_node(
            node_idx, data.x, data.edge_index)

        # Get subgraph from knowledge graph
        center_node = data.node_names[node_idx]
        kg_subgraph = self.kg_constructor.get_subgraph(center_node)

        # Combine GNN explanation with knowledge graph
        combined_explanation = self.combine_explanations(
            node_feat_mask, edge_mask, kg_subgraph)

        return combined_explanation

    def combine_explanations(self, node_feat_mask, edge_mask, kg_subgraph):
        # This is a placeholder for the actual implementation
        # You would need to map the GNN explanation to the knowledge graph entities and relationships
        combined_explanation = {
            'important_features': node_feat_mask.topk(5).indices.tolist(),
            'important_connections': edge_mask.topk(5).indices.tolist(),
            'related_knowledge': [node['name'] for node in kg_subgraph]
        }
        return combined_explanation
