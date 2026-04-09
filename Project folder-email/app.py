import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("spam_new.csv", encoding='latin-1')

# Fix columns
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert label
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['message'] = df['message'].apply(clean_text)

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = df['message']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_spam(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    result = model.predict(vector)
    return "Spam" if result[0] == 1 else "Not Spam"

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📧 Email Spam Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_spam(user_input)
        st.write("Result:", result)
    else:
        st.warning("Please enter a message!")
