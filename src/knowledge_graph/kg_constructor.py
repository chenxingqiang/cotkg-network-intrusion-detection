import uuid
from py2neo import Graph, Node, Relationship
import networkx as nx
import pandas as pd
import numpy as np
import hashlib
import time
import torch
from torch_geometric.data import Data


class KnowledgeGraphConstructor:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="neo4jneo4j", max_retries=3):
        """
        Initialize the KnowledgeGraphConstructor with retry mechanism.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.graph = None
        self.nx_graph = nx.DiGraph()
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j with retry mechanism"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.graph = Graph(self.uri, auth=(self.username, self.password))
                print("Successfully connected to Neo4j database")
                return
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    print(f"Failed to connect to Neo4j after {self.max_retries} attempts: {str(e)}")
                    print("Please ensure Neo4j is running and credentials are correct")
                    print("You can start Neo4j using: brew services start neo4j")
                    raise
                print(f"Connection attempt {retry_count} failed, retrying...")
                time.sleep(2)

    def _convert_properties(self, properties):
        """Convert numpy types to Python native types"""
        if properties is None:
            return {}

        converted = {}
        for k, v in properties.items():
            if isinstance(v, (np.number, np.bool_)):
                converted[k] = float(v)  # Convert all numeric types to float
            elif isinstance(v, np.ndarray):
                converted[k] = v.tolist()
            elif isinstance(v, str):
                converted[k] = v
            else:
                converted[k] = str(v)  # Convert any other types to string
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
        """Add a flow to the knowledge graph"""
        try:
            # Generate consistent node ID
            flow_id = f"Flow_{len(self.nx_graph.nodes())}"
            
            # Add node to NetworkX graph
            self.nx_graph.add_node(flow_id, **flow_data)
            
            # Add edges based on similar features
            for other_node in list(self.nx_graph.nodes()):
                if other_node != flow_id:
                    # Add edge if flows are related
                    if self._are_flows_related(flow_data, self.nx_graph.nodes[other_node]):
                        self.nx_graph.add_edge(flow_id, other_node)
                        self.nx_graph.add_edge(other_node, flow_id)
            
            return True
            
        except Exception as e:
            print(f"Error adding flow: {str(e)}")
            return False

    def _are_flows_related(self, flow1, flow2, similarity_threshold=0.8):
        """Check if two flows are related based on their features"""
        try:
            # Compare numerical features
            common_features = set(flow1.keys()) & set(flow2.keys())
            if not common_features:
                return False
            
            # Calculate similarity score
            similarities = []
            for feature in common_features:
                try:
                    val1 = float(flow1[feature])
                    val2 = float(flow2[feature])
                    # Simple similarity measure
                    similarity = 1 - min(abs(val1 - val2) / max(abs(val1), abs(val2), 1), 1)
                    similarities.append(similarity)
                except (ValueError, TypeError):
                    continue
            
            if not similarities:
                return False
            
            # Return True if average similarity exceeds threshold
            return sum(similarities) / len(similarities) >= similarity_threshold
            
        except Exception as e:
            print(f"Error comparing flows: {str(e)}")
            return False

    def prepare_data_for_graph_sage(self, X, y):
        """
        Prepare data for GraphSAGE model.
        """
        try:
            # Convert features to tensor
            x = torch.FloatTensor(X.values)
            
            # Create node mapping based on X index
            node_mapping = {f"Flow_{i}": i for i in range(len(X))}
            
            # Get edge indices from NetworkX graph
            edge_index = []
            for source, target in self.nx_graph.edges():
                # Only add edges if both nodes are in our mapping
                if source in node_mapping and target in node_mapping:
                    source_idx = node_mapping[source]
                    target_idx = node_mapping[target]
                    # Only add edge if indices are within bounds
                    if source_idx < len(X) and target_idx < len(X):
                        edge_index.append([source_idx, target_idx])
            
            # Add self-loops for all nodes
            for i in range(len(X)):
                edge_index.append([i, i])
            
            # Convert to tensor and transpose
            edge_index = torch.LongTensor(edge_index).t()
            
            # Verify edge indices are within bounds
            max_idx = edge_index.max().item()
            if max_idx >= len(X):
                raise ValueError(f"Edge index {max_idx} is out of bounds for feature matrix of size {len(X)}")
            
            # Convert labels to tensor
            y = torch.LongTensor(y.values)
            
            # Create train/test masks
            num_nodes = x.size(0)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # 80% train, 20% test
            train_size = int(0.8 * num_nodes)
            indices = torch.randperm(num_nodes)
            train_mask[indices[:train_size]] = True
            test_mask[indices[train_size:]] = True
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                test_mask=test_mask
            )
            
            # Verify data is valid
            print("\nGraph data statistics:")
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Number of node features: {data.num_node_features}")
            print(f"Number of classes: {len(torch.unique(data.y))}")
            print(f"Training nodes: {train_mask.sum().item()}")
            print(f"Testing nodes: {test_mask.sum().item()}")
            
            return data
            
        except Exception as e:
            print(f"Error preparing data for GraphSAGE: {str(e)}")
            import traceback
            traceback.print_exc()
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
