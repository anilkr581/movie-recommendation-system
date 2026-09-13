import streamlit as st
import joblib
import requests

st.set_page_config(layout='wide')
st.markdown("""
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
""", unsafe_allow_html=True)
st.sidebar.image('flag.jpg')

st.sidebar.title('About us')
st.sidebar.write('We are a group of ML Engineers and trying to learn NLP')

st.sidebar.title('📞Contact us')
st.sidebar.write('9958966311')

df=joblib.load("dataset.pkl")
X=joblib.load("vectors.pkl")
model=joblib.load("model.pkl")

mv = st.selectbox(
    "Select Movie",
    df['name']
)

if st.button("Recommend"):
    index=df[df.name==mv].index[0]
    mvc=X[index]
    distances,indexes=model.kneighbors([mvc],n_neighbors=5)
    for i in indexes[0][1:]:
        st.write(df.iloc[i]['name'])
        mv_id=df.iloc[i]['movie_id']
        url=f'http://www.omdbapi.com/?i={mv_id}&apikey=b84e8a7c'
        resp=requests.get(url)
        data=resp.json()
        poster_url=data['Poster']
        st.image(poster_url)