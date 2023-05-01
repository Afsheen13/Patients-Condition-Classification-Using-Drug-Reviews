import pandas as  pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


st.title("Patient's Condition Classification")

user_input=st.text_input('Enter the sentences')
sentences=user_input.title()

df=pd.read_csv('drugsCom_raw (1).csv')
df.dropna(axis=0,inplace=True)
df=df.dropna()

df_train = df[(df['condition']=='Depression') | (df['condition']=='High Blood Pressure')|(df['condition']=='Diabetes, Type 2')]

X = df_train.drop(['Unnamed: 0','drugName','rating','date','usefulCount'],axis=1)

# segregating dataframe for analyzing individual condition
X_dep=X[(X['condition']=='Depression')]
X_bp=X[(X['condition']=='High Blood Pressure')]
X_diab=X[(X['condition']=='Diabetes, Type 2')]

for i, col in enumerate(X.columns):
    X.iloc[:, i] = X.iloc[:, i].str.replace('"', '')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup
import re

def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))

X['review_clean'] = X['review'].apply(review_to_words)

X_feat=X['review_clean']
y=X['condition']

X_train, X_test, y_train, y_test = train_test_split(X_feat, y,stratify=y,test_size=0.2, random_state=0)

count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train)

count_test = count_vectorizer.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)

rf_tf = RandomForestClassifier()
rf_tf.fit(tfidf_train_2, y_train)
pred = rf_tf.predict(tfidf_test_2)

## Function for Extracting Top drugs

def top_drugs_extractor(condition):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

tfidf_trigram = tfidf_vectorizer.transform([sentences])


predictions = rf_tf.predict(tfidf_trigram)

if st.button('Predict Condition and Suggest drugs'):
    for text, label in zip(sentences, predictions):
        if label=="High Blood Pressure":
            target="High Blood Pressure"
            top_drugs = top_drugs_extractor(label)
            st.write("\nCondition:", target)
            st.write("Top 3 Suggested Drugs:")
            st.write(top_drugs[0])
            st.write(top_drugs[1])
            st.write(top_drugs[2])
            st.write()
        elif label=="Depression":
            target="Depression"
            top_drugs = top_drugs_extractor(label)
            st.write("\nCondition:", target)
            st.write("Top 3 Suggested Drugs:")
            st.write(top_drugs[0])
            st.write(top_drugs[1])
            st.write(top_drugs[2])
            st.write()
        else: 
            target="Diabetes, Type 2"
            top_drugs = top_drugs_extractor(label)
            st.write("\nCondition:", target)
            st.write("Top 3 Suggested Drugs:")
            st.write(top_drugs[0])
            st.write(top_drugs[1])
            st.write(top_drugs[2])
            st.write()





