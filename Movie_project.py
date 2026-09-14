import joblib
import requests
import streamlit as st

st.set_page_config(layout="wide")
st.markdown(
    """
<div style="
    background-color:#1f4e78;
    padding:20px;
    border-radius:10px;
    text-align:center;
    color:white;
    font-size:30px;
    font-weight:bold;
">
    Movie Recommendation System
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.image("flag.jpg")
st.sidebar.title("About us")
st.sidebar.write("We are a group of ML Engineers and trying to learn NLP")
st.sidebar.title("📞Contact us")
st.sidebar.write("8802587125")

# Load assets
df = joblib.load("dataset.pkl")
X = joblib.load("vectors.pkl")
model = joblib.load("model.pkl")

mv = st.selectbox("Select Movie", df["name"])

# Toggle option for layout
view_mode = st.radio(
    "Select Display Format", ["Grid View", "List View"], horizontal=True
)

if st.button("Recommend"):
  index = df[df.name == mv].index[0]
  mvc = X[index]
  distances, indexes = model.kneighbors([mvc], n_neighbors=5)

  # Fetch recommendation metadata
  recommendations = []
  for i in indexes[0][1:]:
    movie_name = df.iloc[i]["name"]
    mv_id = df.iloc[i]["movie_id"]
    url = f"http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c"
    try:
      resp = requests.get(url)
      data = resp.json()
      poster_url = data.get("Poster", "")
      if poster_url == "N/A" or not poster_url:
        poster_url = "https://via.placeholder.com/300x450?text=No+Poster"
    except Exception:
      poster_url = "https://via.placeholder.com/300x450?text=Error"

    recommendations.append({"name": movie_name, "poster": poster_url})

  # Render based on selected view mode
  if view_mode == "Grid View":
    cols = st.columns(len(recommendations))
    for col, rec in zip(cols, recommendations):
      with col:
        st.image(rec["poster"], use_container_width=True)
        st.markdown(
            f"<p style='text-align: center; font-weight:"
            f" bold;'>{rec['name']}</p>",
            unsafe_allow_html=True,
        )
  else:
    for rec in recommendations:
      col1, col2 = st.columns([1, 4])
      with col1:
        st.image(rec["poster"], width=100)
      with col2:
        st.markdown(
            f"<h4 style='margin-top: 30px;'>{rec['name']}</h4>",
            unsafe_allow_html=True,
        )
      st.divider()