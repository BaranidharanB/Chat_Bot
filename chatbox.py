import random 
import torch
import json
from nltk_code import tokenize,stem
from nltk_code import bag_of_words
from model import NeuralNetwork
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open ("datas.json",'r') as json_datas:
    datas = json.load(json_datas)



FILE = "info_data.pth"
data_info = torch.load(FILE)

input_size = data_info["input_size"]
hidden_size = data_info["hidden_size"]
output_size = data_info["output_size"]
all_words = data_info ["all_words"]
tags = data_info ["tags"]
model_state = data_info ["model_state"]

model = NeuralNetwork(input_size, hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Alexa"
    # Tokenizing
    # IMplementing the model in the chatbot
print("Let's Chat ! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence =="quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model (X)
    _, predicted = torch.max(input=output,dim=1) # this predicts 
    tag = tags[predicted.item()] # This is the class label and the numbers will be the actual tag
    #
    probs = torch.softmax(output,dim = 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # To find the corresponding datas (file) check the matching of the tag
        for data in datas["datas"]:
            if tag == data["tag"]:
                print(f"{bot_name} : {random.choice(data['responses'])}")
    else:
        print (f"{bot_name} : I do not understand ")    




             
