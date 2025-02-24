import unittest
from src.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from src.knowledge_graph.cot_generator import generate_cot, parse_cot_response


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        self.kg_constructor = KnowledgeGraphConstructor(
            'bolt://localhost:7687', 'neo4j', 'password')

    def test_create_entity(self):
        node = self.kg_constructor.create_entity(
            'TestEntity', 'test_node', {'property': 'value'})
        self.assertIsNotNone(node)
        self.assertEqual(node['name'], 'test_node')

    def test_create_relationship(self):

        node1 = self.kg_constructor.create_entity('TestEntity', 'node1')
        node2 = self.kg_constructor.create_entity('TestEntity', 'node2')
        rel = self.kg_constructor.create_relationship(
            node1, 'TEST_REL', node2, {'property': 'value'})
        self.assertIsNotNone(rel)

    def test_add_cot_knowledge(self):
        sample_flow = {
            'Src IP': '192.168.1.1',
            'Dst IP': '10.0.0.1',
            'Protocol': 'TCP',
            'Flow Duration': 1000,
            'Total Fwd Packets': 10,
            'Total Backward Packets': 5
        }
        cot_response = generate_cot(sample_flow)
        entities, relationships = parse_cot_response(cot_response)
        self.kg_constructor.add_cot_knowledge(entities, relationships)

        # Verify that entities and relationships were added
        self.assertGreater(len(self.kg_constructor.graph.nodes), 0)
        self.assertGreater(len(self.kg_constructor.graph.relationships), 0)


if __name__ == '__main__':
    unittest.main()
