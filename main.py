from lab_exam.medical_tfidf_vectorizer import tfidf_embedding,lab_recommendation,similar_health_topics
from utilis.hide_toolbar import hide_toolbars
from utilis.translator import read_text

import streamlit as st

import pickle

#X,vectorizer = np.array([]),TfidfVectorizer()

@st.cache
def embedding():
    X,vectorizer = tfidf_embedding()
    return X,vectorizer

X,vectorizer = embedding()
#hide_toolbars()

st.sidebar.write("""## Fill the form with information about the Health issue of the patient""")

health_issue_input = read_text("Health issue",sidebar = True)

st.title("Lab exam recomendation")
if health_issue_input != "":
    df_lab_rec = lab_recommendation(health_issue_input,X,vectorizer).reset_index(drop=True)
    st.dataframe(df_lab_rec.style.highlight_max(axis=0))

st.title("Similar health topics")
if health_issue_input != "":
    df_similar_ht = similar_health_topics(health_issue_input,X,vectorizer).reset_index(drop=True)
    st.dataframe(df_similar_ht.style.highlight_max(axis=0))