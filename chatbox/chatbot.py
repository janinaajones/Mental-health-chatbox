import json
import pickle
import random

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Emergency keywords
emergency_words = [
    "suicide",
    "kill myself",
    "end my life"
]

def get_response(user_input):
    text = user_input.lower()

    # Emergency check
    for word in emergency_words:
        if word in text:
            return (
                "I'm really sorry you're feeling this way. "
                "Please seek immediate help from a mental health professional "
                "or contact a suicide prevention helpline."
            )

    # Predict intent
    X = vectorizer.transform([text])
    tag = model.predict(X)[0]

    # Find response
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
