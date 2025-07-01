import os
import nltk
import json
import pickle
import random
import numpy as np
import tflearn
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops

# Initialize stemmer
stemmer = LancasterStemmer()

# Load intents
script_directory = os.path.dirname(os.path.realpath(__file__))
intents_path = os.path.join(script_directory, 'intentsOPT.json')
with open(intents_path, 'r') as f:
    intents = json.load(f)

# Prepare training data
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Reset graph
ops.reset_default_graph()

# Build the model architecture (must match original)
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)

# Load or train model
if os.path.exists("model.tflearn.index"):
    print("🔁 Loading existing model...")
    model.load("model.tflearn")
    data = pickle.load(open("training_data", "rb"))
    words = data['words']
    classes = data['classes']
else:
    print("🧠 Training new model...")
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    with open("training_data", "wb") as f:
        pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, f)

# Context management
ERROR_THRESHOLD = 0.25
context = {}

# NLP utility functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Classification
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Get response
def get_response(sentence, userID='123', show_details=False):
    results = classify(sentence)

    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if show_details:
                        print(f"[DEBUG] Predicted tag: {i['tag']}")
                        print(f"[DEBUG] Context before: {context.get(userID)}")

                    # Set context if specified
                    if 'context_set' in i:
                        context[userID] = i['context_set']
                        if show_details:
                            print(f"[DEBUG] Context set to: {context[userID]}")

                    # Check context filter
                    if not 'context_filter' in i or \
                       (userID in context and i['context_filter'] == context[userID]):
                        response = random.choice(i['responses'])
                        if show_details:
                            print(f"[DEBUG] Response: {response}")
                        return response
            results.pop(0)

    return "Sorry, I didn't understand that."

# Optional main chat loop
# if __name__ == "__main__":
#     print("🤖 Chatbot is ready! (type 'quit' to exit)")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == 'quit':
#             break
#         response = get_response(inp, show_details=True)
#         print("Bot:", response)


# This file is for loading trained models
# newPython.py file is used if the model is not even trained for once