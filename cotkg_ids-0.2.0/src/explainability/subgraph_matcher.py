import networkx as nx


class SubgraphMatcher:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def find_matching_subgraphs(self, pattern):
        """
        Find subgraphs in the knowledge graph that match the given pattern.
        :param pattern: A networkx graph representing the pattern to match
        :return: A list of subgraphs that match the pattern
        """
        matcher = nx.algorithms.isomorphism.GraphMatcher(self.kg, pattern,
                                                        node_match=self._node_match,
                                                        edge_match=self._edge_match)
        return list(matcher.subgraph_isomorphisms_iter())

    def _node_match(self, n1, n2):
        """
        Define the criteria for node matching.
        """
        return n1.get('type') == n2.get('type')

    def _edge_match(self, e1, e2):
        """
        Define the criteria for edge matching.
        """
        return e1.get('type') == e2.get('type')

    def explain_detection(self, detection_result, pattern):
        """
        Explain a detection result by finding matching subgraphs in the knowledge graph.
        :param detection_result: The detection result to explain
        :param pattern: A pattern graph related to the detection result
        :return: A list of explanations based on matching subgraphs
        """
        matching_subgraphs = self.find_matching_subgraphs(pattern)
        explanations = []
        for subgraph in matching_subgraphs:
            explanation = self._generate_explanation(
                subgraph, detection_result)
            explanations.append(explanation)
        return explanations

    def _generate_explanation(self, subgraph, detection_result):
        """
        Generate a human-readable explanation based on a matching subgraph.
        """
        # This is a placeholder. The actual implementation would depend on
        # the structure of your knowledge graph and detection results.
        explanation = f"Detection result '{detection_result}' is explained by the following pattern:\n"
        for node in subgraph.nodes(data=True):
            explanation += f"- {node[1]['type']} node: {node[0]}\n"
        for edge in subgraph.edges(data=True):
            explanation += f"- {edge[2]['type']} relationship: {edge[0]} -> {edge[1]}\n"
        return explanation
