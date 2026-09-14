import joblib
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="Movie Recommender")

# --- CUSTOM CSS FOR UNIFORM CARD UI & MULTI-LINE CLAMPING ---
st.markdown(
    """
    <style>
        .movie-card {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            height: 480px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .movie-card img {
            width: 100% !important;
            height: 290px !important;
            object-fit: cover !important;
            border-radius: 6px;
        }
        .movie-info {
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 90px;
        }
        .movie-title {
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .movie-desc {
            font-size: 12px;
            color: #666;
            line-height: 1.4em;
            height: 2.8em;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
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
  # Reliable fallback image data URI or direct public image link to guarantee rendering
  FALLBACK_IMAGE = "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=300&auto=format&fit=crop&q=60"

  for i in indexes[0][1:]:
    name = df.iloc[i]["name"]
    mv_id = df.iloc[i]["movie_id"]
    url = f"http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c"

    poster_url = FALLBACK_IMAGE
    movie_details = {}
    plot_text = "No description available for this movie."
    try:
      resp = requests.get(url, timeout=5)
      data = resp.json()
      if data.get("Response") == "True":
        poster_val = data.get("Poster")
        if poster_val and poster_val != "N/A":
          poster_url = poster_val
        movie_details = data
        if data.get("Plot") and data.get("Plot") != "N/A":
          plot_text = data.get("Plot")
    except Exception:
      pass

    recs.append({
        "name": name,
        "poster": poster_url,
        "plot": plot_text,
        "details": movie_details,
    })

  return recs


# Sidebar elements
st.sidebar.image('flag.jpg')
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

EXCLUDED_FIELDS = {
    "Response",
    "Poster",
    "Ratings",
    "imdbVotes",
    "imdbID",
    "Type",
    "DVD",
    "BoxOffice",
    "Production",
    "Website",
}

# Render View Layouts
if st.session_state.view_mode == "grid":
  cols = st.columns(len(recommendations))
  for col, rec in zip(cols, recommendations):
    with col:
      # Native Streamlit image guarantees container integrity and avoids broken <img> DOM tags
      st.image(rec["poster"], use_container_width=True)

      st.markdown(
          f"""
                <div class="movie-info" style="height: 70px; margin-top: 5px;">
                    <div class="movie-title" title="{rec['name']}">{rec['name']}</div>
                    <div class="movie-desc" title="{rec['plot']}">{rec['plot']}</div>
                </div>
            """,
          unsafe_allow_html=True,
      )

      with st.popover("ℹ️ Details", use_container_width=True):
        st.subheader(rec["name"])
        details = rec["details"]
        if details:
          for key, val in details.items():
            if key not in EXCLUDED_FIELDS:
              st.markdown(f"✔ **{key}**: {val}")
        else:
          st.markdown("✔ **Status**: No extra details available")
else:
  for rec in recommendations:
    with st.container(border=True):
      c1, c2, c3 = st.columns([1, 8, 2])
      with c1:
        st.image(rec["poster"], width=70)
      with c2:
        st.markdown(
            f"""
                <h5 style='margin: 0; padding-top: 5px;'>{rec['name']}</h5>
                <p style='color: #666; font-size: 13px; margin: 4px 0 0 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;'>{rec['plot']}</p>
            """,
            unsafe_allow_html=True,
        )
      with c3:
        st.write("")
        with st.popover("ℹ️ Details", use_container_width=True):
          st.subheader(rec["name"])
          details = rec["details"]
          if details:
            for key, val in details.items():
              if key not in EXCLUDED_FIELDS:
                st.markdown(f"✔ **{key}**: {val}")
          else:
            st.markdown("✔ **Status**: No extra details available")