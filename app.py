import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load your saved model
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

# Load PCA model
@st.cache_resource
def load_pca():
    with open("pca_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
pca = load_pca()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🎬 IMDA Movie Rating Predictor")
st.markdown("Predict the IMDA rating for a movie based on its genres, key cast/crew, and plot embedding.")

# --- Genre dictionary ---
genre_dict = {
    28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime',
    99: 'Documentary', 18: 'Drama', 10751: 'Family', 14: 'Fantasy', 36: 'History',
    27: 'Horror', 10402: 'Music', 9648: 'Mystery', 10749: 'Romance',
    878: 'Science Fiction', 10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
}

# Load the name lists from local CSV files
directors_list = pd.read_csv("directors.csv", header = None).values.flatten().tolist()
actors_list = pd.read_csv("actors.csv", header = None).values.flatten().tolist()
producers_list = pd.read_csv("producers.csv", header = None).values.flatten().tolist()

# clean up any duplicate if exists
directors_list = [name for name in directors_list if not re.search(r'\.\d+$', name)]
actors_list = [name for name in actors_list if not re.search(r'\.\d+$', name)]
producers_list = [name for name in producers_list if not re.search(r'\.\d+$', name)]

# Input fields
selected_directors = st.multiselect("Select directors:", sorted(directors_list))
selected_actors = st.multiselect("Select actors:", sorted(actors_list))
selected_producers = st.multiselect("Select producers:", sorted(producers_list))
genres_input = st.multiselect("Select genres:", options=list(genre_dict.values()), default=["Action", "Thriller"])
subtitles_file = st.file_uploader("Upload subtitles file (.srt):", type=["srt"])

# labels
labels = {
    0: 'PG',
    1: 'NC16',
    2: 'M18',
    3: 'R21'
}

# --- Clean and process .srt subtitles ---
def clean_subtitles_keep_lines(text):
    if pd.isna(text):
        return text
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def split_into_chunks(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_script(text):
    chunks = split_into_chunks(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    return np.mean(embeddings, axis=0)

def one_hot_encoding(data, original_list):
    vector = pd.Series([1 if name in data else 0 for name in original_list], index = original_list)
    df = vector.to_frame().T

    return df

def preprocess_input(directors, actors, producers, genres, subtitles_text):
    cleaned_text = clean_subtitles_keep_lines(subtitles_text)
    if not cleaned_text:
        st.error("Subtitles appear empty after cleaning.")
        return None

    mean_embedding = embed_script(cleaned_text)
    embedding_pca = pca.transform([mean_embedding])[0]
    pca_names = [f'pca_{i}' for i in range(110)]
    embedding_pca_df = pd.Series(embedding_pca, index = pca_names).to_frame().T

    director_df = one_hot_encoding(directors, directors_list)
    actor_df = one_hot_encoding(actors, actors_list)
    producer_df = one_hot_encoding(producers, producers_list)

    original_genre = [f"genres_{name}" for name in genre_dict.values()]
    genre_df = one_hot_encoding(genres, original_genre)

    names_df = director_df.add(actor_df, fill_value=0)
    names_df = names_df.add(producer_df, fill_value=0)
    
    input_df = pd.concat([genre_df, names_df, embedding_pca_df], axis=1)
    return input_df

# Predict button
if st.button("Predict IMDA Rating"):
    if subtitles_file is not None:
        subtitles_text = subtitles_file.read().decode("utf-8")
        input_data = preprocess_input(selected_directors, selected_actors, selected_producers, genres_input, subtitles_text)
        if input_data is not None:
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"🎯 Predicted IMDA Rating: **{labels[prediction]}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Please upload a subtitles file to proceed.")
