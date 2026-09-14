import joblib
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- CUSTOM CSS FOR UNIFORM POSTER SIZES & CARD UI ---
st.markdown(
    """
    <style>
        /* Uniform container for movie poster cards in Grid view */
        .movie-card {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            height: 420px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* Enforce fixed dimensions for images so all boxes match perfectly */
        .movie-card img {
            width: 100% !important;
            height: 330px !important;
            object-fit: cover !important;
            border-radius: 6px;
        }
        .movie-title {
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-top: 8px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
  df = joblib.load("dataset.pkl")
  X = joblib.load("vectors.pkl")
  model = joblib.load("model.pkl")
  return df, X, model


df, X, model = load_models()


@st.cache_data(show_spinner="Fetching recommendations...")
def get_recommendations(movie_name):
  index = df[df.name == movie_name].index[0]
  mvc = X[index]
  distances, indexes = model.kneighbors([mvc], n_neighbors=5)

  recs = []
  # Standard reliable No-Image thumbnail placeholder
  FALLBACK_IMAGE = "https://icon-library.com/images/no-image-icon/no-image-icon-0.jpg"

  for i in indexes[0][1:]:
    name = df.iloc[i]["name"]
    mv_id = df.iloc[i]["movie_id"]
    url = f"http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c"

    poster_url = FALLBACK_IMAGE
    try:
      resp = requests.get(url, timeout=5)
      data = resp.json()
      if (
          data.get("Response") == "True"
          and data.get("Poster")
          and data.get("Poster") != "N/A"
      ):
        poster_url = data.get("Poster")
    except Exception:
      pass

    recs.append({"name": name, "poster": poster_url})

  return recs


# Sidebar elements
st.sidebar.title("🎬 About us")
st.sidebar.write("We are a group of ML Engineers trying to learn NLP.")
st.sidebar.title("📞 Contact us")
st.sidebar.write("8802587125")

# Main Interface
st.title("🍿 Movie Recommendation System")

selected_movie = st.selectbox(
    "Search for a movie to get recommendations:", df["name"]
)

if "view_mode" not in st.session_state:
  st.session_state.view_mode = "grid"

st.divider()

col_title, col_grid, col_list = st.columns([10, 1, 1])
with col_title:
  st.subheader("Recommended for you")
with col_grid:
  if st.button("⊞ Grid", use_container_width=True):
    st.session_state.view_mode = "grid"
with col_list:
  if st.button("☰ List", use_container_width=True):
    st.session_state.view_mode = "list"

recommendations = get_recommendations(selected_movie)

# Render View Layouts
if st.session_state.view_mode == "grid":
  cols = st.columns(len(recommendations))
  for col, rec in zip(cols, recommendations):
    with col:
      st.markdown(
          f"""
                <div class="movie-card">
                    <img src="{rec['poster']}" onerror="this.onerror=null;this.src='https://icon-library.com/images/no-image-icon/no-image-icon-0.jpg';">
                    <div class="movie-title" title="{rec['name']}">{rec['name']}</div>
                </div>
            """,
          unsafe_allow_html=True,
      )
else:
  for rec in recommendations:
    with st.container(border=True):
      c1, c2 = st.columns([1, 10])
      with c1:
        st.image(rec["poster"], width=70)
      with c2:
        st.markdown(
            f"<h5 style='padding-top: 18px; margin: 0;'>{rec['name']}</h5>",
            unsafe_allow_html=True,
        )