import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

st.set_page_config(layout="wide")
st.title("Smth")
st.markdown("Smth")

def create_graph(edges_data):
    graph = nx.DiGraph()
    graph.add_edges_from(edges_data)
    return graph

DEFAULT_EDGES = [
    ('A', 'B'),
    ('B', 'A'),
    ('A', 'C'),
    ('C', 'A'),
    ('C', 'D'),
    ('D', 'E'),
    ('E', 'C'),
    ('F', 'A')
]

def calculate_pagerank(graph):
    if not graph.nodes:
        return {}

    pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)

    st.dataframe(
        {
            "Node": list(pagerank_scores.keys()),
            "PageRank Score": [f"{v:.4f}" for v in pagerank_scores.values()],
        },
        hide_index=True,
        use_container_width=True,
    )
    return pagerank_scores

def visualise_graph(graph, pagerank_scores):
    net = Network(height="600px", width="100%", \
                  bgcolor="#222222", font_color="white", directed=True, notebook=True)
    net.toggle_physics(True)

    scores = list(pagerank_scores.values())
    if not scores: 
        return

    min_pr = min(scores)
    max_pr = max(scores)

    def get_node_size(score):
        if max_pr == min_pr:
            return 30

        normalized = (score - min_pr) / (max_pr - min_pr)
        return 15 + normalized * 35

    for node, score in pagerank_scores.items():
        size = get_node_size(score)
        title = f"PageRank: {score:.4f}"

        net.add_node(
            n_id=node,
            label=node,
            size=size,
            title=title,
            color="#A0CBE2"
        )

    for source, target in graph.edges():
        net.add_edge(source, target, arrows='to', color='gray')

    temp_filename = "pyvis_temp_graph.html"

    net.show(temp_filename)

    with open(temp_filename, 'r') as f:
        html_content = f.read()

    components.html(html_content, height=620)

def main():
    st.markdown("---")
    st.header("Write your edges")
    edge_input = st.text_area(
        "Enter edges like 'A,B'",
        value="\n".join([f"{s},{t}" for s, t in DEFAULT_EDGES]),
        height=200
    )

    recalculate_button = st.button("Recalculate", type="primary")

    user_edges = []
    if edge_input:
        for line in edge_input.strip().split('\n'):
            if line:
                source, target = [x.strip() for x in line.split(',')]
                user_edges.append((source, target))

    graph = create_graph(user_edges)

    st.header("PageRank Scores")
    pagerank_scores = calculate_pagerank(graph)


    st.header("Interactive Graph Visualization")

    visualise_graph(graph, pagerank_scores)

if __name__ == '__main__':
    main()
