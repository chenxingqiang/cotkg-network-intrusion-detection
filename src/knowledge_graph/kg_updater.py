from py2neo import Graph, NodeMatcher, RelationshipMatcher


class KnowledgeGraphUpdater:
    def __init__(self, graph):
        self.graph = graph
        self.node_matcher = NodeMatcher(graph)
        self.rel_matcher = RelationshipMatcher(graph)

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
