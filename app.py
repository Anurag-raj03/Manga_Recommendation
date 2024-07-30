import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib as j
from PIL import Image
import requests
from io import BytesIO
import random

# Load pre-processed data
main_df = j.load('main_df.jbl')
personal_df = j.load('personal_df.jbl')
matched_value = j.load('/mnt/data/matched_value.pkl')

# Define the recommendation function
def mangas_suggest(manga_name):
    index = np.where(main_df.index == manga_name)[0][0]
    similar_manga = sorted(list(enumerate(matched_value[index])), key=lambda x: x[1], reverse=True)[1:10]
    search_manga = []
    for i in similar_manga:
        mang = []
        temp_df = personal_df[personal_df['Name'] == main_df.index[i[0]]]
        mang.extend(list(temp_df.drop_duplicates('Name')['Name']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Genre']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Rating']))
        mang.extend(list(temp_df.drop_duplicates('Name')['img-link']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Link']))
        search_manga.append(mang)
    return search_manga

# Function to get random manga
def random_manga(n=9):
    all_manga = list(main_df.index)
    random.shuffle(all_manga)
    random_manga_list = all_manga[:n]
    search_manga = []
    for manga in random_manga_list:
        mang = []
        temp_df = personal_df[personal_df['Name'] == manga]
        mang.extend(list(temp_df.drop_duplicates('Name')['Name']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Genre']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Rating']))
        mang.extend(list(temp_df.drop_duplicates('Name')['img-link']))
        mang.extend(list(temp_df.drop_duplicates('Name')['Link']))
        search_manga.append(mang)
    return search_manga

# Streamlit App
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stImage img {
        border-radius: 8px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        width: 150px;
        height: 200px;
    }
    .stImage img:hover {
        transform: scale(1.05);
    }
    .manga-name {
        font-size: 20px;
        color: #ff4b4b;
        font-weight: bold;
        text-align: center;
    }
    .manga-genre, .manga-rating, .manga-link {
        font-size: 16px;
        color: #007bff;
        text-align: center;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
    }
    .card {
        background-color: #fff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        width: 150px;
        margin: 10px;
    }
    .card:hover {
        transform: scale(1.05);
    }
    .card-content {
        text-align: center;
    }
    .read-button a {
        background-color: #007bff;
        color: white;
        padding: 10px;
        text-decoration: none;
        border-radius: 5px;
        transition: 0.3s;
        display: inline-block;
        margin-top: 10px;
    }
    .read-button a:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Manga Recommendation System</div>', unsafe_allow_html=True)

# User input for manga name with searchable selectbox
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("")

with col2:
    manga_name = st.selectbox(
        'Select a Manga',
        options=[''] + list(main_df.index),
        format_func=lambda x: 'Select a Manga' if x == '' else x
    )

    search_button = st.button('Search', key='search_button')

with col3:
    st.write("")

if search_button and manga_name:
    recommendations = mangas_suggest(manga_name)
    st.markdown(f"<div class='title'>Recommendations for {manga_name}:</div>", unsafe_allow_html=True)
else:
    recommendations = random_manga()

# Display manga recommendations in a grid
st.markdown("<div class='container'>", unsafe_allow_html=True)
for rec in recommendations:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    response = requests.get(rec[3])
    img = Image.open(BytesIO(response.content))
    st.image(img, caption=rec[0], use_column_width=True)
    st.markdown(f"<div class='card-content'>", unsafe_allow_html=True)
    st.markdown(f"<div class='manga-name'>{rec[0]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='manga-genre'>**Genre**: {', '.join(rec[1])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='manga-rating'>**Rating**: {rec[2]}</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="read-button"><a href="{rec[4]}" target="_blank">Read Manga</a></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Ensure proper padding and margin between each manga entry
st.markdown(
    """
    <style>
    .stImage {
        padding: 10px;
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Run the app with `streamlit run your_script.py`
