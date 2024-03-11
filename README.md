# Chatbot-using-Tensorflow-and-Flask

This repository contains code for a simple chatbot implemented using Flask and TFlearn. The chatbot is trained on intents specified in a JSON file and can respond to user queries based on the trained model.

## Features

- User-friendly chat interface
- Handles various user queries using trained intents
- Can be easily integrated into web applications

## Prerequisites

- Python 3.x installed on your system
- Flask and TFlearn libraries installed (`pip install flask tflearn tensorflow`)

## Setup and Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd chatbot
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Access the chat interface in your web browser at `http://localhost:5000`.

5. Start chatting with the chatbot by entering your queries in the input field.

## Structure

- `app.py`: Contains the Flask application code.
- `intents.json`: JSON file containing intents for training the chatbot.
- `model.tflearn`: Trained model saved using TFlearn.
- `training_data`: Pickled file containing training data.

