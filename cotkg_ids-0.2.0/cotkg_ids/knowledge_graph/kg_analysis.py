import pandas as pd
from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class KnowledgeGraphAnalyzer:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="neo4jneo4j"):
        self.graph = Graph(uri, auth=(username, password))
    
    def get_graph_statistics(self):
        """Get basic statistics about the knowledge graph"""
        stats = {}
        
        # Count nodes by type
        stats['nodes'] = {}
        for label in ['Flow', 'Feature', 'Attack']:
            count = self.graph.evaluate(f"MATCH (n:{label}) RETURN count(n)")
            stats['nodes'][label] = count
        
        # Count relationships by type
        stats['relationships'] = {}
        for rel_type in ['HAS_FEATURE', 'HAS_ATTACK']:
            count = self.graph.evaluate(f"MATCH ()-[r:{rel_type}]->() RETURN count(r)")
            stats['relationships'][rel_type] = count
        
        return stats
    
    def get_attack_distribution(self):
        """Get distribution of attack types"""
        query = """
        MATCH (a:Attack)
        RETURN a.value as attack_type, count(*) as count
        ORDER BY count DESC
        """
        result = self.graph.run(query).data()
        return pd.DataFrame(result)
    
    def get_feature_statistics(self):
        """Get statistics about features"""
        query = """
        MATCH (f:Feature)
        RETURN f.name as feature_name, 
               count(*) as frequency,
               avg(toFloat(f.value)) as avg_value,
               min(toFloat(f.value)) as min_value,
               max(toFloat(f.value)) as max_value
        ORDER BY frequency DESC
        """
        result = self.graph.run(query).data()
        return pd.DataFrame(result)
    
    def visualize_graph_sample(self, limit=50):
        """Visualize a sample of the knowledge graph"""
        try:
            # Get a sample of the graph
            query = f"""
            MATCH (f:Flow)-[r]->(n)
            WITH f, r, n LIMIT {limit}
            RETURN f, r, n
            """
            result = self.graph.run(query).data()
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for record in result:
                flow = record['f']
                target = record['n']
                rel = record['r']
                
                # Add nodes with proper attributes
                flow_attrs = dict(flow)
                flow_attrs['node_type'] = 'Flow'  # Use node_type instead of label
                G.add_node(flow.identity, **flow_attrs)
                
                target_attrs = dict(target)
                target_attrs['node_type'] = list(target.labels)[0]  # Get first label
                G.add_node(target.identity, **target_attrs)
                
                # Add edge
                G.add_edge(flow.identity, target.identity, 
                          edge_type=type(rel).__name__)
            
            # Visualize
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(G)
            
            # Draw nodes by type
            node_colors = {
                'Flow': 'lightblue',
                'Feature': 'lightgreen',
                'Attack': 'lightcoral'
            }
            
            for node_type in ['Flow', 'Feature', 'Attack']:
                nodes = [n for n, d in G.nodes(data=True) 
                        if d.get('node_type') == node_type]
                nx.draw_networkx_nodes(G, pos, 
                                     nodelist=nodes,
                                     node_color=node_colors[node_type],
                                     node_size=500,
                                     label=node_type)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, 
                                 edge_color='gray',
                                 arrows=True)
            
            # Add labels
            labels = {node: f"{data['node_type']}\n{data.get('name', '')[:10]}"
                     for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title("Knowledge Graph Sample Visualization")
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('knowledge_graph_sample.png')
            plt.close()
            
        except Exception as e:
            print(f"Error in graph visualization: {str(e)}")
            import traceback
            traceback.print_exc()

def analyze_knowledge_graph():
    """Analyze and report on the knowledge graph"""
    analyzer = KnowledgeGraphAnalyzer()
    
    # 1. Get basic statistics
    print("\nKnowledge Graph Statistics:")
    stats = analyzer.get_graph_statistics()
    print("\nNode counts:")
    for node_type, count in stats['nodes'].items():
        print(f"  {node_type}: {count:,}")
    print("\nRelationship counts:")
    for rel_type, count in stats['relationships'].items():
        print(f"  {rel_type}: {count:,}")
    
    # 2. Get attack distribution
    print("\nAttack Type Distribution:")
    attack_dist = analyzer.get_attack_distribution()
    print(attack_dist)
    
    # 3. Get feature statistics
    print("\nFeature Statistics:")
    feature_stats = analyzer.get_feature_statistics()
    print(feature_stats)
    
    # 4. Visualize graph sample
    print("\nGenerating graph visualization...")
    analyzer.visualize_graph_sample()
    print("Graph visualization saved as 'knowledge_graph_sample.png'")

if __name__ == "__main__":
    analyze_knowledge_graph() 