import networkx as nx
import plotly.graph_objs as go


def visualize_knowledge_graph(graph, center_node=None, depth=2):
    if center_node:
        subgraph = nx.ego_graph(graph, center_node, radius=depth)
    else:
        subgraph = graph

    pos = nx.spring_layout(subgraph)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    for node in subgraph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(subgraph.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = f'{node}<br># of connections: {len(adjacencies[1])}'
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()


def visualize_feature_importance(feature_importance):
    features, importances = zip(*feature_importance)

    fig = go.Figure([go.Bar(x=features, y=importances)])
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score'
    )
    fig.show()
