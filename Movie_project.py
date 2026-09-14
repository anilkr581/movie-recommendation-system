import joblib
import requests
import streamlit as st

st.set_page_config(layout="wide")

# App Header Styling
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

# Sidebar info
st.sidebar.title("About us")
st.sidebar.write("We are a group of ML Engineers and trying to learn NLP")
st.sidebar.title("📞Contact us")
st.sidebar.write("8802587125")

# Load models and dataset
df = joblib.load("dataset.pkl")
X = joblib.load("vectors.pkl")
model = joblib.load("model.pkl")

mv = st.selectbox("Select Movie", df["name"])

# Initialize session state for view mode ('grid' or 'list')
if "view_mode" not in st.session_state:
  st.session_state.view_mode = "grid"

# Row container for layout switch icons only
col_spacer, col_grid_icon, col_list_icon = st.columns([10, 1, 1])

with col_grid_icon:
  if st.button("⊞", help="Switch to Grid View"):
    st.session_state.view_mode = "grid"

with col_list_icon:
  if st.button("☰", help="Switch to List View"):
    st.session_state.view_mode = "list"

# Fallback image icon if poster is missing or unavailable
FALLBACK_IMAGE = "https://cdn-icons-png.flaticon.com/512/2748/2748558.png"

if st.button("Recommend"):
  index = df[df.name == mv].index[0]
  mvc = X[index]
  distances, indexes = model.kneighbors([mvc], n_neighbors=5)

  recommendations = []
  for i in indexes[0][1:]:
    movie_name = df.iloc[i]["name"]
    mv_id = df.iloc[i]["movie_id"]
    url = f"http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c"

    poster_url = FALLBACK_IMAGE
    try:
      resp = requests.get(url, timeout=5)
      data = resp.json()
      if data.get("Response") == "True" and data.get("Poster") != "N/A":
        poster_url = data.get("Poster", FALLBACK_IMAGE)
    except Exception:
      poster_url = FALLBACK_IMAGE

    recommendations.append({"name": movie_name, "poster": poster_url})

  # Save results to session state so view switching doesn't wipe recommendations out
  st.session_state.recommendations = recommendations

# Render recommendations if they exist in session state
if "recommendations" in st.session_state:
  recs = st.session_state.recommendations

  st.write("### Recommended Movies:")

  if st.session_state.view_mode == "grid":
    # Grid View Layout (Equal columns based on number of recommendations)
    cols = st.columns(len(recs))
    for col, rec in zip(cols, recs):
      with col:
        st.image(rec["poster"], use_container_width=True)
        st.markdown(
            f"<p style='text-align: center; font-size: 14px;'><b>{rec['name']}</b></p>",
            unsafe_allow_html=True,
        )
  else:
    # List View Layout (Compact side-by-side card format)
    for rec in recs:
      c1, c2 = st.columns([1, 8])
      with c1:
        st.image(rec["poster"], width=70)
      with c2:
        st.markdown(
            f"<h4 style='margin: 0; padding-top: 15px;'>{rec['name']}</h4>",
            unsafe_allow_html=True,
        )
      st.divider()