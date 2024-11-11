import uuid
from py2neo import Graph, Node, Relationship
import networkx as nx
import pandas as pd
import numpy as np
import hashlib


class KnowledgeGraphConstructor:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="neo4jneo4j"):
        """
        Initialize the KnowledgeGraphConstructor.
        """
        self.graph = Graph(uri, auth=(username, password))
        self.nx_graph = nx.DiGraph()

    def _convert_properties(self, properties):
        """Convert numpy types to Python native types"""
        if properties is None:
            return {}

        converted = {}
        for k, v in properties.items():
            if isinstance(v, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
                converted[k] = int(v)
            elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
                converted[k] = float(v)
            elif isinstance(v, np.bool_):
                converted[k] = bool(v)
            elif isinstance(v, np.ndarray):
                converted[k] = v.tolist()
            else:
                converted[k] = v
        return converted

    def add_node(self, node_type, node_id, properties=None):
        """
        Add a node to both Neo4j and NetworkX graphs.
        """
        try:
            # Convert properties to native Python types
            properties = self._convert_properties(properties or {})

            # Remove 'name' from properties if it exists to avoid duplicate
            properties.pop('name', None)

            # Create Neo4j node with node_id as name property
            node = Node(node_type, **properties)
            node['name'] = node_id  # Add name after node creation

            self.graph.create(node)

            # Add to NetworkX graph
            self.nx_graph.add_node(node_id,
                                 node_type=node_type,
                                 **properties)
            return node
        except Exception as e:
            print(f"Error adding node: {str(e)}")
            return None

    def add_edge(self, source_id, edge_type, target_id, properties=None):
        """
        Add an edge to both Neo4j and NetworkX graphs.
        """
        try:
            # Convert properties to native Python types
            properties = self._convert_properties(properties)

            # Get source and target nodes from Neo4j
            source_node = self.graph.nodes.match(name=source_id).first()
            target_node = self.graph.nodes.match(name=target_id).first()

            if source_node and target_node:
                # Create Neo4j relationship
                rel = Relationship(source_node, edge_type, target_node, **properties)
                self.graph.create(rel)

                # Add to NetworkX graph
                self.nx_graph.add_edge(source_id, target_id,
                                     edge_type=edge_type,
                                     **properties)
                return rel
        except Exception as e:
            print(f"Error adding edge: {str(e)}")
            return None

    def add_flow(self, flow_data):
        """Add flow data to knowledge graph"""
        try:
            # Ensure data is dictionary format
            if not isinstance(flow_data, dict):
                flow_data = dict(flow_data)

            # Convert numpy types to Python native types
            flow_data = self._convert_properties(flow_data)

            # Handle NaN values
            flow_data = {k: ('null' if pd.isna(v) else v) for k, v in flow_data.items()}

            # Generate unique flow ID
            flow_features = [
                str(flow_data.get('flow_duration', '')),
                str(flow_data.get('Protocol Type', '')),
                str(flow_data.get('Rate', ''))
            ]

            # Generate flow_id using MD5
            feature_string = '_'.join(str(f) for f in flow_features if f)
            flow_id = hashlib.md5(feature_string.encode()).hexdigest()[:8]

            # Add Flow node
            flow_node = self.add_node(
                'Flow',
                f"Flow_{flow_id}",
                flow_data
            )

            if not flow_node:
                return

            # Add Attack node and relationship
            label = flow_data.get('label', 'Unknown')
            attack_node = self.add_node('Attack', f"Attack_{label}", {'value': label})
            if attack_node:
                self.add_edge(f"Flow_{flow_id}", 'HAS_ATTACK', f"Attack_{label}")

            # Add Feature nodes and relationships
            for feature_name, feature_value in flow_data.items():
                if feature_name == 'label' or pd.isna(feature_value):
                    continue

                feature_id = f"Feature_{feature_name}_{feature_value}"
                feature_node = self.add_node(
                    'Feature',
                    feature_id,
                    {'name': feature_name, 'value': feature_value}
                )
                if feature_node:
                    self.add_edge(f"Flow_{flow_id}", 'HAS_FEATURE', feature_id)

        except Exception as e:
            print(f"Error adding flow to knowledge graph: {str(e)}")
            raise

    def prepare_data_for_graph_sage(self, X, y):
        """
        Prepare data for GraphSAGE model.
        """
        try:
            # Convert NetworkX graph to PyTorch Geometric data
            # This is a placeholder - implement based on your specific needs
            from torch_geometric.utils import from_networkx

            # Add node features to NetworkX graph
            for idx, row in X.iterrows():
                node_id = f"Flow_{idx}"
                if node_id in self.nx_graph:
                    self.nx_graph.nodes[node_id]['features'] = row.values

            # Convert to PyTorch Geometric data
            data = from_networkx(self.nx_graph)
            data.y = y  # Add labels

            return data

        except Exception as e:
            print(f"Error preparing data for GraphSAGE: {str(e)}")
            return None

    def clear_graph(self):
        """
        Clear both Neo4j and NetworkX graphs.
        """
        try:
            self.graph.delete_all()
            self.nx_graph.clear()
        except Exception as e:
            print(f"Error clearing graph: {str(e)}")
