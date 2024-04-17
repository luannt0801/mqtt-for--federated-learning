import numpy as np
import pandas as pd
import random

import os
import torch
import json
import string
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from collections import OrderedDict

from tldextract import extract

from tqdm import tqdm
from datetime import datetime
import time

import matplotlib.pyplot as plt
from glob_inc.utils import *

NUM_ROUND = 50
ROUND_DICT = {}
batch_size = 64
lr = 8e-6

num_file = 150
############################################################################################################

max_features = 101 # max_features = number of one-hot dimensions
embed_size = 64
hidden_size = 64
n_layers = 1

maxlen = 127
char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
ix2char = {ix:char for char, ix in char2ix.items()}


def load_data(df):
    """
        Input pandas DataFrame
        Output DataLoader
    """
    max_features = 101 # max_features = number of one-hot dimensions
    maxlen = 127
    batch_size = 64

    domains = df['domain'].to_numpy()
    labels = df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]
    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)
    trainset = TensorDataset(torch.tensor(padded_domains, dtype=torch.long), torch.Tensor(encoded_labels))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader

def domain2tensor(domains):
    encoded_domains = [[char2ix[y] for y in domain] for domain in domains]
    padded_domains = pad_sequences(encoded_domains, maxlen)
    tensor_domains = torch.LongTensor(padded_domains)
    return tensor_domains

def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0]*(maxlen-len(domain))+domain)
    return np.asarray(domains)

def decision(x):
    return x >= 0.5

def save_dataframe(client_id = 1):
    data_folder = 'validation_data/'
    dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, dga_type))]
    #print(dga_types)
    my_df = pd.DataFrame(columns=['domain', 'type', 'label'])
    for dga_type in dga_types:
        files = os.listdir(os.path.join(data_folder, dga_type))
        for file in files:
            with open(os.path.join(data_folder, dga_type, file), 'r') as fp:
                domains_with_type = [[(line.strip()), dga_type, 1] for line in fp.readlines()[:num_file]]
                appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                my_df = pd.concat([my_df, appending_df], ignore_index=True)
                
    with open(os.path.join(data_folder, 'benign.txt'), 'r') as fp:
        domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:num_file*10*len(dga_types)]]
        appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
        my_df = pd.concat([my_df, appending_df], ignore_index=True)
        
    print(my_df['label'].value_counts())
    return my_df

def split_train_test_data(final_df):
    train_test_df, val_df = train_test_split(final_df, test_size=0.10, shuffle=True) 
    #print(train_test_df)
    # Pre-processing
    domains = train_test_df['domain'].to_numpy()
    labels = train_test_df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]

    #print(f"Number of samples: {len(encoded_domains)}")
    #print(f"One-hot dims: {len(char2ix) + 1}")
    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)

    #X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)
    #test toan bo
    X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)


    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainloader, testloader

class LSTMModel(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, n_layers):
        super(LSTMModel, self).__init__()
        
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(feat_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x, hidden):
        embedded_feats = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded_feats, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        fc_out = self.fc(lstm_out)
        sigmoid_out = self.sigmoid(fc_out)
        sigmoid_out = sigmoid_out.view(x.shape[0], -1)
        sigmoid_last = sigmoid_out[:,-1]

        return sigmoid_last, hidden
    
    def init_hidden(self, x):
        weight = next(self.parameters()).data
        h = (weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_(),
             weight.new(self.n_layers, x.shape[0], self.hidden_size).zero_())
        return h
    
    def get_embeddings(self, x):
        return self.embedding(x)

def test(model, testloader, criterion, batch_size):
    val_h = model.init_hidden(domain2tensor(["0"]*batch_size))
    model.eval()
    eval_losses = []
    total = 0
    correct = 0
    # Initialize TP, FP, TN, FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():

        for eval_inputs, eval_labels in tqdm(testloader):
            
            eval_inputs = eval_inputs.to(device)
            eval_labels = eval_labels.to(device)
            
            val_h = tuple([x.data for x in val_h])
            eval_output, val_h = model(eval_inputs, val_h)
            
            eval_prediction = decision(eval_output)
            total += len(eval_prediction)
            correct += sum(eval_prediction == eval_labels)
            
            # Update TP, FP, TN, FN
            for pred, label in zip(eval_prediction, eval_labels):
                if pred == 1 and label == 1:
                    TP += 1
                elif pred == 1 and label == 0:
                    FP += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 0 and label == 1:
                    FN += 1
            
            eval_loss = criterion(eval_output.squeeze(), eval_labels.float())
            eval_losses.append(eval_loss.item())
    
    # Calculate precision and recall
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    accuracy_check = (TP+TN)/total

    return np.mean(eval_losses), correct/total, TP, FP, TN, FN, precision, recall , accuracy_check


############################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(max_features, embed_size, hidden_size, n_layers).to(device)

#thay the tham so truyen vào ứng với tập test để đánh gia mô hình
print_log("load data ...")
my_df = save_dataframe()
trainloader, testloader = split_train_test_data(my_df)

#print(len(trainloader))
#print(len(testloader))

def do_evaluate_round():
    #duyệt qua tất cả các round để đánh giá kết quả mỗi round
    for round_idx in range(NUM_ROUND):
        print(f"\nEvaluate Round {round_idx + 1}:\n")
        model_path = f"./model_round_{round_idx + 1}.pt"
        #load model từ model aggregate
        model.load_state_dict(torch.load(model_path, map_location=device))

        criterion = nn.BCELoss(reduction='mean')
        optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

        eval_loss, accuracy, TP, FP, TN, FN, precision, recall , accuracy_check = test(model=model, testloader=testloader, criterion=criterion, batch_size=batch_size)
        print(
            "Eval Loss: {:.4f}".format(eval_loss),
            "Accuracy: {:.4f}".format(accuracy),
            "TP: {}".format(TP),
            "FP: {}".format(FP),
            "TN: {}".format(TN),
            "FN: {}".format(FN),
            "Precision: {:.4f}".format(precision),
            "Recall: {:.4f}".format(recall),
            "Accuracy check: {:.4f}".format(accuracy_check)
        )

        ROUND_DICT[f"round_{round_idx + 1}"] = {
            "accuracy": accuracy,
            "eval_loss": eval_loss,
            "TP" :(TP),
            "FP" :(FP),
            "TN" :(TN),
            "FN" : (FN),
            "Precision" : (precision),
            "Recall" : (recall),
        }
    print(ROUND_DICT)


#start_training_task()
if __name__ == "__main__":
    do_evaluate_round()
    # Extract accuracy values from round_dict
    accuracies = [ROUND_DICT[f"round_{i+1}"]["accuracy"] for i in range(NUM_ROUND)]

    # Extract accuracy and avg_loss values from round_dict
    accuracies = [ROUND_DICT[f"round_{i+1}"]["accuracy"] for i in range(NUM_ROUND)]
    avg_losses = [ROUND_DICT[f"round_{i+1}"]["eval_loss"] for i in range(NUM_ROUND)]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, NUM_ROUND + 1), accuracies, marker='o')
    axs[0].set_title('Accuracy over rounds')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_xticks(range(1, NUM_ROUND + 1))
    axs[0].grid(True)

    # Plot average loss
    axs[1].plot(range(1, NUM_ROUND + 1), avg_losses, marker='o', color='red')
    axs[1].set_title('Average Loss over rounds')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Average Loss')
    axs[1].set_xticks(range(1, NUM_ROUND + 1))
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()