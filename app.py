import streamlit as st
from ann_index import ANNIndexer
from utils import load_vectors, get_embedder, encode_text

# Load data and model once
@st.cache_resource
def setup_app(method):
    vectors = load_vectors()
    indexer = ANNIndexer(vectors, method=method)
    model = get_embedder()
    return indexer, model, vectors

st.set_page_config(page_title="ANN Search App", layout="wide")

st.title("🔍 Approximate Nearest Neighbor Search")

method = st.selectbox("Choose ANN method:", ["Flat", "HNSW", "IVFPQ"])
indexer, model, vectors = setup_app(method)

query = st.text_input("Enter your query (e.g., 'cat', 'machine learning', etc.)")

if query:
    query_vec = encode_text(query, model)
    distances, indices = indexer.search(query_vec, k=5)

    st.subheader("🔗 Top 5 Nearest Neighbors")
    for rank, (i, d) in enumerate(zip(indices, distances), 1):
        st.markdown(f"**#{rank}** — Vector ID: `{i}`, Distance: `{d:.4f}`")
