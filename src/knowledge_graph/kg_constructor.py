import uuid
from py2neo import Graph, Node, Relationship
import networkx as nx
import pandas as pd
import numpy as np
import hashlib
import time
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from .cot_generator import generate_cot, parse_cot_response


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
        """Add a flow to the knowledge graph with COT analysis"""
        try:
            # Generate consistent node ID
            flow_id = f"Flow_{len(self.nx_graph.nodes())}"
            
            # Format flow data for COT analysis
            flow_str = "\n".join([f"{k}: {v}" for k, v in flow_data.items()])
            
            # Generate COT analysis
            try:
                cot_response = generate_cot(flow_str)
                entities, relationships = parse_cot_response(cot_response)
                
                # Add COT-derived entities
                for entity in entities:
                    entity_id = f"{entity['type']}_{hashlib.md5(entity['name'].encode()).hexdigest()[:8]}"
                    self.add_node(entity['type'], entity_id, {'name': entity['name']})
                    
                    # Connect flow to entity
                    if entity['type'] == 'Attack':
                        self.add_edge(flow_id, 'CLASSIFIED_AS', entity_id)
                    else:
                        self.add_edge(flow_id, 'HAS_FEATURE', entity_id)
                
                # Add COT-derived relationships
                for rel in relationships:
                    source_id = f"Feature_{hashlib.md5(rel['source'].encode()).hexdigest()[:8]}"
                    target_id = f"Attack_{hashlib.md5(rel['target'].encode()).hexdigest()[:8]}"
                    self.add_edge(source_id, rel['type'], target_id)
                    
            except Exception as e:
                print(f"Error in COT analysis: {str(e)}")
            
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
        Prepare data for GraphSAGE model with optimized graph construction.
        """
        try:
            # Convert features to tensor
            x = torch.FloatTensor(X.values)
            
            # Create node mapping based on X index
            node_mapping = {f"Flow_{i}": i for i in range(len(X))}
            
            # Get edge indices from NetworkX graph
            edge_index = []
            
            # Add edges based on feature similarity
            similarity_matrix = cosine_similarity(X)
            
            # Add edges for similar nodes
            for i in range(len(X)):
                # Get top k similar nodes
                k = min(20, len(X) - 1)  # Adjust k based on dataset size
                similar_indices = similarity_matrix[i].argsort()[-k:]
                
                for j in similar_indices:
                    if i != j:  # Avoid self-loops for now
                        edge_index.append([i, j])
            
            # Add edges based on label similarity
            for i in range(len(X)):
                # Connect to nodes with same label
                same_label_indices = y.loc[y == y[i]].index
                for j in same_label_indices[:10]:  # Limit to 10 same-label connections
                    if i != j:
                        edge_index.append([i, j])
            
            # Add self-loops with special attention
            for i in range(len(X)):
                # Add multiple self-loops to emphasize self-information
                for _ in range(3):  # Add 3 self-loops per node
                    edge_index.append([i, i])
            
            # Convert to tensor and transpose
            edge_index = torch.LongTensor(edge_index).t()
            
            # Remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
            
            # Verify edge indices are within bounds
            max_idx = edge_index.max().item()
            if max_idx >= len(X):
                raise ValueError(f"Edge index {max_idx} is out of bounds for feature matrix of size {len(X)}")
            
            # Convert labels to tensor
            y = torch.LongTensor(y.values)
            
            # Create stratified train/test split masks
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            
            num_nodes = x.size(0)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            for train_idx, test_idx in sss.split(X, y):
                train_mask[train_idx] = True
                test_mask[test_idx] = True
            
            # Create validation mask from training set
            val_size = int(0.1 * num_nodes)  # 10% validation set
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_indices = train_mask.nonzero().view(-1)
            val_indices = train_indices[:val_size]
            train_mask[val_indices] = False
            val_mask[val_indices] = True
            
            # Create PyG Data object with validation mask
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            # Print statistics
            print("\nOptimized Graph data statistics:")
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Number of node features: {data.num_node_features}")
            print(f"Number of classes: {len(torch.unique(data.y))}")
            print(f"Training nodes: {train_mask.sum().item()}")
            print(f"Validation nodes: {val_mask.sum().item()}")
            print(f"Testing nodes: {test_mask.sum().item()}")
            print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
            
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
