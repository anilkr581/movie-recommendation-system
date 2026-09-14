import joblib
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- 1. CACHE MODELS AND DATA ---
# This prevents reloading large files every time the user interacts with the app
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    df = joblib.load("dataset.pkl")
    X = joblib.load("vectors.pkl")
    model = joblib.load("model.pkl")
    return df, X, model

df, X, model = load_models()

# --- 2. CACHE API CALLS ---
# This ensures toggling between list/grid doesn't re-trigger slow API fetches
@st.cache_data(show_spinner="Fetching recommendations...")
def get_recommendations(movie_name):
    index = df[df.name == movie_name].index[0]
    mvc = X[index]
    distances, indexes = model.kneighbors([mvc], n_neighbors=5)

    recs = []
    FALLBACK_IMAGE = "https://cdn-icons-png.flaticon.com/512/2748/2748558.png"

    for i in indexes[0][1:]:
        name = df.iloc[i]["name"]
        mv_id = df.iloc[i]["movie_id"]
        url = f"http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c"

        poster_url = FALLBACK_IMAGE
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get("Response") == "True" and data.get("Poster") != "N/A":
                poster_url = data.get("Poster")
        except Exception:
            pass # Keep fallback image on failure

        recs.append({"name": name, "poster": poster_url})
    
    return recs

# --- 3. SIDEBAR ---
st.sidebar.title("🎬 About us")
st.sidebar.write("We are a group of ML Engineers trying to learn NLP.")
st.sidebar.title("📞 Contact us")
st.sidebar.write("8802587125")

# --- 4. MAIN UI ---
st.title("🍿 Movie Recommendation System")

# Dropdown triggers updates automatically (no button needed)
selected_movie = st.selectbox("Search for a movie to get recommendations:", df["name"])

# Initialize View State
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "grid"

st.divider()

# Controls Layout (Title on left, view toggles on right)
col_title, col_grid, col_list = st.columns([10, 1, 1])
with col_title:
    st.subheader("Recommended for you")
with col_grid:
    if st.button("⊞ Grid", use_container_width=True):
        st.session_state.view_mode = "grid"
with col_list:
    if st.button("☰ List", use_container_width=True):
        st.session_state.view_mode = "list"

# Fetch data for the currently selected movie
recommendations = get_recommendations(selected_movie)

# --- 5. DISPLAY RENDERER ---
if st.session_state.view_mode == "grid":
    # Clean Grid View
    cols = st.columns(len(recommendations))
    for col, rec in zip(cols, recommendations):
        with col:
            st.image(rec["poster"], use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-weight: 500;'>{rec['name']}</p>", unsafe_allow_html=True)
else:
    # Clean List View using borders
    for rec in recommendations:
        with st.container(border=True):
            c1, c2 = st.columns([1, 10])
            with c1:
                st.image(rec["poster"], width=60)
            with c2:
                # Vertical alignment spacing for list text
                st.markdown(f"<h5 style='padding-top: 15px;'>{rec['name']}</h5>", unsafe_allow_html=True)