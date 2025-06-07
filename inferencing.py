import streamlit as st
import pickle
from modelling import NetflixRecommender

@st.cache_resource
def load_model():
    with open("netflix_df.pkl", "rb") as f:
        df = pickle.load(f)
        model = NetflixRecommender("dummy.csv")  
    model.df = df
    model.build_model()
    return model

recommender = load_model()

st.title("🎬 Netflix Content-Based Recommender")
selected_title = st.text_input("Enter movie/tv shows:")

if st.button("Show Recommedation") and selected_title:
    result = recommender.get_recommendations(selected_title, topn=5)
    if isinstance(result, str):
        st.warning(result)  # 
    else:
        st.success(f"You might like this also")
        for _, row in result.iterrows():
            st.markdown(f"### 🎥 {row['title']} ({row['release_year']})")
            st.markdown(f"**Type**: {row['type']}  \n**Genre**: {', '.join(row['genres'])}")
            st.markdown(f"_{row['description']}_")
            st.markdown("---")


