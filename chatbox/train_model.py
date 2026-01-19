import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intent data
with open("intents.json", "r") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# Train ML model
model = LogisticRegression()
model.fit(X, labels)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully")

