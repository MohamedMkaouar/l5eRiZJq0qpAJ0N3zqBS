import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

health_topics = pickle.load(open('resources/health_topics.pickle','rb'))
medical_tests = pickle.load(open('resources/medical_tests.pickle','rb'))

def tfidf_embedding(health_topics = health_topics,medical_tests = medical_tests):
    corpus = list(health_topics.values())+list(np.array(list(medical_tests.values()))[:,0])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X,vectorizer


def lab_recommendation(text,X,vectorizer):

    nbr_health_topics = len(list(health_topics.keys()))
    df = pd.DataFrame(X.T.toarray(),columns = list(health_topics.keys())+list(medical_tests.keys()))
    vector = vectorizer.transform([text]).T.toarray()
    score = []
    for i,c in enumerate(df.columns):
        if i >= nbr_health_topics:
            score.append(cosine_similarity(vector.reshape(1, -1),df[[c]].to_numpy().reshape(1, -1))[0,0])
    best_matches = pd.DataFrame( {'Lab exam': list(df.columns)[nbr_health_topics:],'score':score}).sort_values(by="score",ascending=False)

    return(best_matches)


def similar_health_topics(text,X,vectorizer):

    nbr_health_topics = len(list(health_topics.keys()))
    df = pd.DataFrame(X.T.toarray(),columns = list(health_topics.keys())+list(medical_tests.keys()))
    vector = vectorizer.transform([text]).T.toarray()
    score = []
    for i,c in enumerate(df.columns):
        if i < nbr_health_topics:
            score.append(cosine_similarity(vector.reshape(1, -1),df[[c]].to_numpy().reshape(1, -1))[0,0])
    best_matches = pd.DataFrame( {'Health topics': list(df.columns)[:nbr_health_topics],'score':score}).sort_values(by="score",ascending=False)

    return(best_matches)