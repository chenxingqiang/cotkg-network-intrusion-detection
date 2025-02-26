import unittest
import time
from cotkg_ids.knowledge_graph.kg_constructor import KnowledgeGraphConstructor
from cotkg_ids.knowledge_graph.cot_generator import generate_cot, parse_cot_response


class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.kg_constructor = KnowledgeGraphConstructor(
                    'bolt://localhost:7687', 'neo4j', 'neo4j')  # Using default password
                # Test connection
                self.kg_constructor.graph.run("MATCH (n) RETURN n LIMIT 1")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.skipTest(f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}\n"
                                "Please ensure Neo4j is running and credentials are correct\n"
                                "You can start Neo4j using: brew services start neo4j")
                print(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)

    def test_create_entity(self):
        """Test entity creation."""
        try:
            entity = {
                'id': 'test_entity',
                'type': 'Attack',
                'properties': {'name': 'Test Attack', 'severity': 'high'}
            }
            result = self.kg_constructor.create_entity(entity)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"create_entity failed: {str(e)}")

    def test_create_relationship(self):
        """Test relationship creation."""
        try:
            source = {
                'id': 'source_entity',
                'type': 'Feature',
                'properties': {'name': 'Source Feature'}
            }
            target = {
                'id': 'target_entity',
                'type': 'Attack',
                'properties': {'name': 'Target Attack'}
            }
            relationship = {
                'type': 'INDICATES',
                'properties': {'confidence': 0.9}
            }
            
            # Create entities first
            self.kg_constructor.create_entity(source)
            self.kg_constructor.create_entity(target)
            
            # Create relationship
            result = self.kg_constructor.create_relationship(
                source['id'], target['id'], relationship)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"create_relationship failed: {str(e)}")

    def test_add_cot_knowledge(self):
        """Test adding CoT knowledge to the graph."""
        try:
            cot_text = "This is a test CoT response"
            entities, relationships = parse_cot_response(cot_text)
            
            # Add knowledge to graph
            for entity in entities:
                result = self.kg_constructor.create_entity(entity)
                self.assertIsNotNone(result)
            
            for rel in relationships:
                result = self.kg_constructor.create_relationship(
                    rel['source'], rel['target'], 
                    {'type': rel['type'], 'properties': {}}
                )
                self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"add_cot_knowledge failed: {str(e)}")

    def tearDown(self):
        """Clean up test data."""
        if hasattr(self, 'kg_constructor'):
            try:
                # Delete test data
                self.kg_constructor.graph.run(
                    "MATCH (n) WHERE n.id IN ['test_entity', 'source_entity', 'target_entity'] DETACH DELETE n"
                )
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == '__main__':
    unittest.main()
