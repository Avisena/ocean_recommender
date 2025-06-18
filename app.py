import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Load Data ---
@st.cache_data
def load_data():
    import kagglehub
    dataset_path = kagglehub.dataset_download('utkarshshrivastav07/career-prediction-dataset')

    # Find the CSV file
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                return pd.read_csv(os.path.join(root, file))

    return None

# --- Similarity Finder ---
def find_similar_careers(df, input_scores, top_n=5):
    ocean_cols = ['O_score', 'C_score', 'E_score', 'A_score', 'N_score']
    df = df.dropna(subset=ocean_cols)
    features = df[ocean_cols].values

    input_vector = np.array(input_scores).reshape(1, -1)
    similarities = cosine_similarity(input_vector, features)[0]
    top_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            'Career': df.iloc[idx]['Career'],
            'O_score': df.iloc[idx]['O_score'],
            'C_score': df.iloc[idx]['C_score'],
            'E_score': df.iloc[idx]['E_score'],
            'A_score': df.iloc[idx]['A_score'],
            'N_score': df.iloc[idx]['N_score'],
            'Similarity': round(similarities[idx], 4)
        })
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("🔍 Career Match by Personality (OCEAN Model)")

st.markdown("Enter your OCEAN personality scores (range: 1.0 – 10.0):")

O = st.number_input("Openness (O)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
C = st.number_input("Conscientiousness (C)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
E = st.number_input("Extraversion (E)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
A = st.number_input("Agreeableness (A)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
N = st.number_input("Neuroticism (N)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

top_n = st.slider("How many recommendations?", 1, 10, 5)

if st.button("🔎 Find Similar Careers"):
    st.info("Loading data and computing similarities...")
    df = load_data()
    if df is not None:
        input_scores = [O, C, E, A, N]
        result_df = find_similar_careers(df, input_scores, top_n)
        st.success("Here are the most similar personalities and their careers:")
        st.dataframe(result_df)
    else:
        st.error("❌ Failed to load dataset.")
