import json
from nltk_code import tokenize,stem
from nltk_code import bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)



# Reading the Json file with r and loading it
with open('datas.json','r') as f: 
    datas = json.load(f)

all_words = []
tags = []
xy = [] # Hold both pattern and tags

for data in datas['datas']:
    tag = data['tag']
    tags.append(tag) # appending every tags to it

    for pattern in data['patterns']:
        w = tokenize (pattern)
        all_words.extend(w) # We're using extend is to expand the existing array to avoid array of arrays
        xy.append((w,tag)) #xy will hold both pattern and tags

ignore_words = ['?','!','.',',']
# Applying the stemming in the all_words by ignoring the exceptions
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Sorting the words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = [] # Bag of words
Y_train = [] # tags of associate numbers

for (pattern_sentence, tag) in xy:
    #X: bag of words for each pattern_sentence
    bag = bag_of_words (pattern_sentence,all_words)
    X_train.append(bag)
    # Y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    Y_train.append(label) # Crossentropy loss


X_train = np.array(X_train)
Y_train = np.array(Y_train)

# This is to train the data by iterating
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# Hyper Parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len (X_train[0])
learning_rate = 0.001
n_epochs = 1000
# print(input_size,len(all_words))
# print(output_size,tags)

#
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork (input_size, hidden_size, output_size ).to(device)


#loss and optimizer

criteron = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# Training Loop
for epoch in range (n_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype = torch.long).to(device)

        #Forward
        outputs = model(words)
        loss = criteron(outputs,labels)

        #backward and Optimzer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'epoch {epoch + 1}/{n_epochs},loss ={loss.item():.4f} ')

print(f'final loss,loss ={loss.item():.4f} ')

info_data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags

}

FILE = "info_data.pth"
torch.save(info_data,FILE)

print(f'Training Completed. File saved to {FILE}')