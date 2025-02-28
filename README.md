# Personal Medical Assistant Chatbot

This project is a Personal Medical Assistant chatbot built using deep learning techniques. The chatbot leverages BERT for natural language understanding and TensorFlow/Keras for intent classification. Below is a guide on how to set up and run the project.

## Table of Contents
1. Project Overview
2. Requirements
3. Setup Instructions
4. Code Explanation
5. Running the Project
6. Usage
7. License

## Project Overview
The Personal Medical Assistant chatbot provides personalized medical advice based on user queries. Users can interact with the chatbot to receive guidance on various medical issues.


## Requirements
- Python 3.7 or higher
- Required libraries:
    - numpy
    - random
    - json
    - pickle
    - torch
    - gradio
    - transformers
    - tensorflow
    - nltk

You can install the required libraries using pip:
```bash
pip install -r requirements.txt
```

## Setup Instructions
- Clone the Repository:
```bash
git clone https://github.com/kaybrian/Chatbot_ML_T1.git
```
- Navigate to the cloned directory:
```bash
cd Chatbot_ML_T1
```
- Run the following command to install all the required libraries:
```bash
pip install -r requirements.txt
```

## Code Explanation
- Importing Libraries
    - The required libraries are imported at the beginning of the script.

- Loading Files
    - The trained model, words, classes, and intents data are loaded using pickle and json.

- BERT Tokenizer and Model
    - BERT tokenizer and model are initialized to get embeddings for user input.

- Functions
    - get_bert_embedding(sentence): Computes BERT embeddings for the input sentence.
    - predict_class(sentence): Predicts the intent of the user input using the trained model.
    - get_response(intents_list, intents_json): Retrieves a random response based on the predicted intent.
    - chatbot_response(message): Main function that integrates prediction and response retrieval.

##  Gradio Interface

A Gradio interface is created for user interaction with the chatbot.

## Running the Project

To run the chatbot, use the following command in the terminal:

```bash
python app.py
```

## Usage

- Once the Gradio interface is launched, a web page will open in your default browser.
- Type your medical query into the text box and press "Enter" or click the submit button.
- The chatbot will respond with relevant medical advice or information.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Authors
- [Kayongo Johnson Brian](https://github.com/kaybrian) 