import numpy as np
import random
import json
import pickle
import torch
import gradio as gr
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Load necessary files
model = load_model('./data/model/chatbotmodel.h5')
words = pickle.load(open('./data/model/words.pkl', 'rb'))
classes = pickle.load(open('./data/model/classes.pkl', 'rb'))
with open("./data/chat/data.json", "r") as json_file:
    dict_ = json.load(json_file)

# Load the BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
lemmatizer = WordNetLemmatizer()

# Function to get BERT embeddings
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to predict the intent of a user's input
def predict_class(sentence):
    embedding = get_bert_embedding(sentence)
    res = model.predict(embedding)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# Function to get the response based on the intent predicted
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if tag in intent['tags']:
                return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# Function that Gradio will use to provide chatbot responses
def chatbot_response(message):
    intents = predict_class(message)
    response = get_response(intents, dict_)
    return response

# Enhanced CSS for a modern, visually appealing interface
css = """
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chatbot-title {
    font-size: 3em;
    text-align: center;
    margin-bottom: 0.5em;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}
.gr-chatbot {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 1em;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    max-height: 500px;
    overflow-y: auto;
}
.gr-textbox {
    border-radius: 25px;
    padding: 10px;
    border: none;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}
.gr-button {
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1em;
    background-color: #ffffff;
    color: #764ba2;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}
.gr-button:hover {
    background-color: #f1f1f1;
}
.clear-button {
    background-color: #e74c3c !important;
    color: #ffffff !important;
    border-radius: 25px;
    padding: 10px 20px;
    font-size: 1em;
}
.clear-button:hover {
    background-color: #c0392b !important;
}
"""

# Create a Gradio Blocks-based interface with custom styling
with gr.Blocks(theme="huggingface", css=css, title="Personal Medical Assistant") as demo:
    gr.Markdown("<h1 class='chatbot-title'>💬 Personal Medical Assistant Chatbot</h1>")
    
    # Chat component for the conversation
    chatbot = gr.Chatbot(elem_classes="gr-chatbot")
    
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type your message here...", lines=1, elem_classes="gr-textbox")
        send = gr.Button("Send", variant="primary", elem_classes="gr-button")
        
    clear = gr.Button("Clear Chat", variant="secondary", elem_classes="clear-button")
    
    # Function to update the chat history
    def respond(message, chat_history):
        if message:
            bot_response = chatbot_response(message)
            chat_history = chat_history + [[message, bot_response]]
        return "", chat_history

    # Bind the submit and click events
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.launch()
