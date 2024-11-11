from py2neo import Graph, Node, Relationship
import networkx as nx
import pandas as pd
import numpy as np
import hashlib


class KnowledgeGraphUpdater:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="neo4jneo4j"):
        """
        Initialize the KnowledgeGraphUpdater.
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

    def update_knowledge(self, entities, relationships):
        for entity in entities:
            self._update_entity(entity)

        for relationship in relationships:
            self._update_relationship(relationship)

    def _update_entity(self, entity):
        existing_node = self.node_matcher.match(
            entity['type'], name=entity['name']).first()
        if existing_node:
            # Update properties
            existing_node.update(entity.get('properties', {}))
            self.graph.push(existing_node)
        else:
            # Create new node
            new_node = Node(
                entity['type'], name=entity['name'], **entity.get('properties', {}))
            self.graph.create(new_node)

    def _update_relationship(self, relationship):
        start_node = self.node_matcher.match(
            name=relationship['source']).first()
        end_node = self.node_matcher.match(name=relationship['target']).first()

        if start_node and end_node:
            existing_rel = self.rel_matcher.match(
                (start_node, end_node), r_type=relationship['type']).first()
            if existing_rel:
                # Update properties
                existing_rel.update(relationship.get('properties', {}))
                self.graph.push(existing_rel)
            else:
                # Create new relationship
                new_rel = Relationship(
                    start_node, relationship['type'], end_node, **relationship.get('properties', {}))
                self.graph.create(new_rel)

    def prune_outdated_knowledge(self, threshold_date):
        # Remove nodes and relationships that haven't been updated recently
        query = f"""
        MATCH (n)
        WHERE n.last_updated < $threshold
        DETACH DELETE n
        """
        self.graph.run(query, threshold=threshold_date)
