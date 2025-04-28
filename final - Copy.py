import streamlit as st
import faiss
import pickle
import numpy as np
import os
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel, CLIPProcessor, CLIPModel

# Set Streamlit page configuration
st.set_page_config(page_title="Content Based Apparel Recommendation System")

# File paths
metadata_file_path = r"C:\Users\anish\Downloads\final_year_demo\metadata.pkl"
tfidf_cosine_index_file_path = "faiss_index_cosine_3000.idx"
tfidf_euclidean_index_file_path = "faiss_index_euclidean_3000.idx"
tfidf_vectorizer_file_path = "tfidf_vectorizer_3000.pkl"
glove_dict_file_path = r"C:\Users\anish\Downloads\final_year_demo\glove.pkl"
vit_index_path = r"C:\Users\anish\Desktop\Project\faiss_index_vit.bin"
clip_index_path = r"C:\Users\anish\Desktop\Project\faiss_index_clip.bin"
image_paths_file = r"C:\Users\anish\Desktop\Project\faiss_index_vit.bin.paths"

# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <h3 style='font-size:24px; color: #084d9a;'><b><u>Karunya Institute of Technology and Sciences</u></b></h3>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.image("karunya_logo.jpg", width=100)

# Load metadata and vectorizer
@st.cache_data
def load_metadata_and_vectorizer():
    with open(metadata_file_path, "rb") as f:
        metadata = pickle.load(f)
    with open(tfidf_vectorizer_file_path, "rb") as vec_file:
        tfidf_vectorizer = pickle.load(vec_file)
    return metadata, tfidf_vectorizer

# Load FAISS indices
@st.cache_data
def load_faiss_indices():
    tfidf_cosine_index = faiss.read_index(tfidf_cosine_index_file_path)
    tfidf_euclidean_index = faiss.read_index(tfidf_euclidean_index_file_path)
    vit_index = faiss.read_index(vit_index_path)
    clip_index = faiss.read_index(clip_index_path)
    return tfidf_cosine_index, tfidf_euclidean_index, vit_index, clip_index

# Load image paths
@st.cache_data
def load_image_paths():
    with open(image_paths_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths

# Function to search and display images
def search_and_display_image(uploaded_image, selected_index, feature_extractor, vit_model, clip_processor, clip_model,
                             image_paths, metadata, method, top_n=5):
    st.markdown(f"<h4 style='color: #084d9a;'>Uploaded Image</h4>", unsafe_allow_html=True)
    st.image(uploaded_image, caption="Uploaded Image", width=200)

    query_image = Image.open(uploaded_image).convert("RGB")
    if method == "ViT":
        inputs = feature_extractor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)
        query_features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    elif method == "CLIP":
        inputs = clip_processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        query_features = outputs.squeeze().numpy()

    query_features = query_features.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_features)

    distances, indices = selected_index.search(query_features, top_n)
    retrieved_images = [(image_paths[int(idx)], metadata[int(idx)], dist) for idx, dist in zip(indices[0], distances[0])]

    if not retrieved_images:
        st.error("No similar images found. Please try another image.")
        return

    st.markdown(f"<h4 style='color: #084d9a;'>Similar Images ({method})</h4>", unsafe_allow_html=True)
    for img_path, meta, dist in retrieved_images:
        similarity_score = 1 - dist
        title = meta.get("title", "N/A")
        asin = meta.get("asin", "N/A")

        st.image(img_path, width=150)
        st.markdown(f"**Title**: {title}")
        st.markdown(f"**ASIN**: {asin}")
        st.markdown(f"**Similarity Score**: {similarity_score:.4f}")
        st.markdown("---")

# Sidebar options
search_type = st.sidebar.selectbox("Search Apparels Based on..", ("Text", "Image"))
top_k = st.sidebar.slider("Select number of results (Top K)", min_value=1, max_value=10, value=5)

# Load FAISS indices once at the start
tfidf_cosine_index, tfidf_euclidean_index, vit_index, clip_index = load_faiss_indices()

if search_type == "Text":
    text_model = st.sidebar.radio("Choose Text Model:", ("TF-IDF", "GloVe"))
    metric = st.sidebar.radio("Choose Distance Metric:", ("Cosine", "Euclidean"))
    query_text = st.text_input("Enter your search query")

    if query_text:
        metadata, tfidf_vectorizer = load_metadata_and_vectorizer()
        selected_index = tfidf_cosine_index if metric == "Cosine" else tfidf_euclidean_index
        query_vector = tfidf_vectorizer.transform([query_text]).toarray().astype("float32")

        faiss.normalize_L2(query_vector)
        distances, indices = selected_index.search(query_vector, top_k)

        st.markdown(f"<h4 style='color: #084d9a;'>Search Results ({metric.capitalize()} Similarity)</h4>",
                    unsafe_allow_html=True)
        for dist, idx in zip(distances[0], indices[0]):
            result = metadata[idx]
            title = result.get("title", "N/A")
            asin = result.get("asin", "N/A")
            image_path = result.get("image_path", "")

            st.markdown(f"**Title**: {title}")
            st.markdown(f"**ASIN**: {asin}")
            st.markdown(f"**Similarity Score**: {1 - dist:.4f}")
            if os.path.exists(image_path):
                st.image(image_path, width=150)
            else:
                st.markdown("<span style='color: red;'>Image not available</span>", unsafe_allow_html=True)
            st.markdown("---")

if search_type == "Image":
    model = st.sidebar.radio("Choose Image Model:", ("ViT", "CLIP"))
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        metadata, _ = load_metadata_and_vectorizer()
        search_and_display_image(
            uploaded_file,
            vit_index if model == "ViT" else clip_index,
            ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224"),
            ViTModel.from_pretrained("google/vit-base-patch16-224"),
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
            load_image_paths(),
            metadata,
            model,
            top_k
        )