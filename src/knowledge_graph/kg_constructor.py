from py2neo import Graph, Node, Relationship
import networkx as nx


class KnowledgeGraphConstructor:
    def __init__(self, uri, username, password):
        self.graph = Graph(uri, auth=(username, password))
        self.nx_graph = nx.DiGraph()

    def create_entity(self, entity_type, name, properties=None):
        node = Node(entity_type, name=name)
        if properties:
            node.update(properties)
        self.graph.create(node)
        self.nx_graph.add_node(name, entity_type=entity_type, **properties)
        return node

    def create_relationship(self, source_node, rel_type, target_node, properties=None):
        rel = Relationship(source_node, rel_type, target_node)
        if properties:
            rel.update(properties)
        self.graph.create(rel)
        self.nx_graph.add_edge(source_node['name'], target_node['name'],
                               rel_type=rel_type, **properties)
        return rel

    def add_flow(self, flow_data):
        flow_node = self.create_entity(
            'Flow', f"Flow_{flow_data['Flow ID']}", flow_data)

        # Add relationships based on flow characteristics
        if flow_data['Label'] != 'BENIGN':
            attack_node = self.create_entity('Attack', flow_data['Label'])
            self.create_relationship(flow_node, 'IS_INSTANCE_OF', attack_node)

        # Add more relationships based on flow characteristics
        if flow_data['Dst Port'] == 80:
            service_node = self.create_entity('Service', 'HTTP')
            self.create_relationship(flow_node, 'USES', service_node)

        # Add more complex relationships and entities as needed

    def prepare_data_for_graph_sage(self, X, y):
        # Convert knowledge graph to PyTorch Geometric data
        # This is a placeholder and needs to be implemented based on your specific needs
        pass


    def add_cot_knowledge(self, entities, relationships):
        for entity in entities:
            self.create_entity(
                entity['type'], entity['name'], entity.get('properties'))

        for rel in relationships:
            source = self.graph.nodes.match(name=rel['source']).first()
            target = self.graph.nodes.match(name=rel['target']).first()
            if source and target:
                self.create_relationship(
                    source, rel['type'], target, rel.get('properties'))
