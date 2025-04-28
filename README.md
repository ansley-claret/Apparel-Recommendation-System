# Apparel-Recommendation-System

📌 Project Overview
This project implements a Content-Based Apparel Recommendation System that allows users to search for clothing products using both text queries and image uploads. The system integrates Natural Language Processing (NLP) and Computer Vision (CV) techniques to deliver personalized, real-time fashion recommendations.

It uses:

TF-IDF, Word2Vec, GloVe for text vectorization

ViT (Vision Transformer) and CLIP (Contrastive Language–Image Pretraining) for image feature extraction

FAISS (Facebook AI Similarity Search) for fast and scalable similarity search

Streamlit for a fully interactive web-based user interface.

⚙️ Features
📄 Text-based Search: Enter keywords to find matching apparel items.

🖼️ Image-based Search: Upload an apparel image to retrieve visually similar products.

🔥 Multimodal Flexibility: Supports multiple vectorization techniques and similarity metrics.

⚡ Real-time Recommendations: Results displayed instantly with similarity scores.

🎨 User-friendly Streamlit Interface: Simple, fast, and highly interactive.


🛠️ Technologies Used
Python 3.8+

Streamlit

FAISS (for vector similarity search)

PyTorch and Hugging Face Transformers (for ViT and CLIP models)

Scikit-learn (for TF-IDF vectorizer)

Pickle (for saving/loading metadata and indices)

NumPy, PIL, OS

├── metadata.pkl
├── faiss_index_cosine_3000.idx
├── faiss_index_euclidean_3000.idx
├── tfidf_vectorizer_3000.pkl
├── glove.pkl
├── faiss_index_vit.bin
├── faiss_index_clip.bin
├── app.py (Streamlit App)
├── README.md
├── requirements.txt
└── images/
