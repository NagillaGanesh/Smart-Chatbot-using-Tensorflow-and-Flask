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

stemmer = LancasterStemmer()

# Get path of current script
script_directory = os.path.dirname(os.path.realpath(__file__))
intents_file_path = os.path.join(script_directory, 'intentsOPT.json')

# Load intents JSON
with open(intents_file_path, 'r') as json_file:
    intents = json.load(json_file)

# NLP preprocessing
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

# Create training data
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Reset TF graph
ops.reset_default_graph()

# Build model
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# Save training data
with open("training_data", "wb") as f:
    pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, f)

# Load model and data
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']

model.load('./model.tflearn')

# NLP functions
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

# Inference and context
ERROR_THRESHOLD = 0.25
context = {}

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

def get_response(sentence, userID='123', show_details=False):
    results = classify(sentence)

    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    # Debug: show predicted tag and context state
                    if show_details:
                        print(f"[DEBUG] Predicted tag: {i['tag']}")
                        print(f"[DEBUG] Context before: {context.get(userID)}")

                    # Set context if available
                    if 'context_set' in i:
                        context[userID] = i['context_set']
                        if show_details:
                            print(f"[DEBUG] Context set to: {context[userID]}")

                    # Check context filter
                    if 'context_filter' not in i or \
                       (userID in context and i['context_filter'] == context[userID]):
                        response = random.choice(i['responses'])
                        if show_details:
                            print(f"[DEBUG] Returning response: {response}")
                        return response
            results.pop(0)

    return "Sorry, I didn't understand that."

# Optional CLI loop
# if __name__ == "__main__":
#     print("Bot is ready! (type 'quit' to exit)")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == 'quit':
#             break
#         print("Bot:", get_response(inp, show_details=True))

# This file is for training
# newPythonSpare.py file is used if the model is trained for once and the models are saved in the current directory