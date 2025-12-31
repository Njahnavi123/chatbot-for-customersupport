import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("dataset/chatbot_data.csv")

X = data["question"]
y = data["answer"]

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
with open("model/chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Chatbot model trained and saved successfully.")
