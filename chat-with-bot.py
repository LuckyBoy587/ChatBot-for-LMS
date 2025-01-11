import json
import nltk
import string
import numpy as np
import pickle
from keras.src.saving import load_model

lemmatizer = nltk.stem.WordNetLemmatizer()
ignore_words = string.punctuation
intents = json.load(open('data/intents.json', 'r'))["intents"]
unique_words = pickle.load(open('data/words.pkl', 'rb'))
unique_classes = pickle.load(open('data/classes.pkl', 'rb'))
model = load_model('data/chatbot_model.keras')


def input_parser(word: str):
    word_list = nltk.word_tokenize(word)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_words]
    features = [0] * len(unique_words)
    for word in word_list:
        if word in unique_words:
            features[unique_words.index(word)] = 1
    return np.array([features])


while True:
    user_input = input("You: ")
    prediction = model.predict(input_parser(user_input))
    print(f"Prediction probability: {np.max(prediction)}")
    class_index = np.argmax(prediction)
    tag = unique_classes[class_index]
    print(f"Predicted class: {tag}")
    for intent in intents:
        if intent["tag"] == tag:
            responses = intent["responses"]
            print(f"Bot: {np.random.choice(responses)}")
            break
    if tag == "goodbye":
        break