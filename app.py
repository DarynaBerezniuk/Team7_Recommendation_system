import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import pagerank_calculation as pg
import random

st.set_page_config(layout="wide")
st.title("Dating app UCU")

def create_graph(edges_data):
    graph = nx.DiGraph()
    graph.add_edges_from(edges_data)
    return graph

def visualise_graph(graph, pagerank_scores):
    net = Network(height="600px", width="100%", bgcolor="#222222",
                  font_color="white", directed=True, notebook=True)
    net.toggle_physics(True)

    if not pagerank_scores:
        return

    min_pr, max_pr = min(pagerank_scores.values()), max(pagerank_scores.values())

    def get_node_size(score):
        if max_pr == min_pr:
            return 30
        normalized = (score - min_pr) / (max_pr - min_pr)
        return 15 + normalized * 35

    for node, score in pagerank_scores.items():
        net.add_node(node, label=node, size=get_node_size(score),
                     title=f"PageRank: {score:.4f}", color="#A0CBE2")

    for source, target in graph.edges():
        net.add_edge(source, target, arrows='to', color='gray')

    temp_filename = "pyvis_temp_graph.html"
    net.save_graph(temp_filename)
    with open(temp_filename, 'r', encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=620)

def pagerank_calculation(file_likes='Likes.ini'):
    likes = pg.graph_creation(file_likes)
    users = pg.get_all_users(likes)
    matrix = pg.create_transition_matrix(likes, users)
    if not users:
        st.warning("У графі немає жодного користувача.")
        return
    
    if "liked" not in st.session_state: st.session_state.liked = []
    if "disliked" not in st.session_state: st.session_state.disliked = []
    if "asked" not in st.session_state: st.session_state.asked = []
    if "current_candidate" not in st.session_state:
        st.session_state.current_candidate = 'name_18'  # same as console

    if len(st.session_state.asked) == len(users):
        st.success("Ми показали всі доступні профілі!")
        st.write("Твої лайки:", st.session_state.liked or "немає")
        st.write("Твої дизлайки:", st.session_state.disliked or "немає")
        return

    if st.session_state.current_candidate in st.session_state.asked:
        remaining = [u for u in users if u not in st.session_state.asked]
        if not remaining:
            st.success("Ми показали всі доступні профілі!")
            st.write("Твої лайки:", st.session_state.liked or "немає")
            st.write("Твої дизлайки:", st.session_state.disliked or "немає")
            return
        st.session_state.current_candidate = remaining[0]

    st.subheader("Поточний кандидат")
    st.write(pg.suggest_people(st.session_state.current_candidate, 'hobbies.ini'))

    acted = False
    col1, col2 = st.columns(2)
    if col1.button("👍 Подобається"):
        if st.session_state.current_candidate not in st.session_state.liked:
            st.session_state.liked.append(st.session_state.current_candidate)
        st.session_state.asked.append(st.session_state.current_candidate)
        acted = True
    if col2.button("👎 Не подобається"):
        if st.session_state.current_candidate not in st.session_state.disliked:
            st.session_state.disliked.append(st.session_state.current_candidate)
        st.session_state.asked.append(st.session_state.current_candidate)
        acted = True

    if acted and not st.session_state.liked and not st.session_state.disliked:
        remaining = [u for u in users if u not in st.session_state.asked]
        if not remaining:
            st.success("Ми показали всі доступні профілі 🙂")
            st.write("Твої лайки:", st.session_state.liked or "немає")
            st.write("Твої дизлайки:", st.session_state.disliked or "немає")
            return
        st.session_state.current_candidate = random.choice(remaining)

    clan_users = pg.build_clan_from_likes(likes, st.session_state.liked, st.session_state.disliked)
    extended_disliked = pg.extend_disliked_with_neighbors(likes, st.session_state.disliked)
    personalization = pg.build_personalization_vector(
        users=users,
        liked_users=st.session_state.liked,
        disliked_users=list(extended_disliked),
        clan_users=clan_users,
    )
    pageranks = pg.calculate_personalized_pagerank(matrix, personalization)
    pagerank_scores = dict(zip(users, pageranks))

    recommendations = pg.generate_recommendations(
        users,
        pageranks,
        current_user=None,
        liked_users=st.session_state.liked,
        disliked_users=st.session_state.disliked,
        asked_users=st.session_state.asked,
    )

    if acted and (st.session_state.liked or st.session_state.disliked):
        if not recommendations:
            st.success("Ми показали всі доступні профілі 🙂")
            st.write("Твої лайки:", st.session_state.liked or "немає")
            st.write("Твої дизлайки:", st.session_state.disliked or "немає")
            return
        next_candidates = [name for name, _ in recommendations if name not in st.session_state.asked]
        if next_candidates:
            st.session_state.current_candidate = next_candidates[0]
        else:
            remaining = [u for u in users if u not in st.session_state.asked]
            st.session_state.current_candidate = remaining[0] if remaining else None
    edges_data = [(src, tgt) for src, tgts in likes.items() for tgt in tgts]

    graph = create_graph(edges_data)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Рекомендації")
        st.dataframe({"User": [r[0] for r in recommendations], "Score": [f"{r[1]:.4f}" for r in recommendations]}, \
                hide_index=True)

    with col2:
        st.subheader("Граф")
        visualise_graph(graph, pagerank_scores)

    st.write("DEBUG:",
         {"current": st.session_state.current_candidate,
          "liked": st.session_state.liked,
          "disliked": st.session_state.disliked,
          "asked": st.session_state.asked,
          "top_recs": recommendations[:5]})


if __name__ == '__main__':
    pagerank_calculation()
