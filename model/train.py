import torch
from torch import autograd, nn
import torch.nn.functional as F
import pandas as pd
from numpy import genfromtxt
import time

import data
import model as m

###############################################################################
# Training Parameters
###############################################################################

input_size = 3150
hidden_size = 3
num_classes = 839
batch_size = 11
learning_rate = .001
epochs = 10
bptt = 100

###############################################################################
# Load data
###############################################################################

train_src = torch.from_numpy(genfromtxt('../data/_final/train_src.csv', delimiter=','))
train_tgt = torch.from_numpy(genfromtxt('../data/_final/train_tgt.csv', delimiter=','))
val_src = torch.from_numpy(genfromtxt('../data/_final/val_src.csv', delimiter=','))
val_tgt = torch.from_numpy(genfromtxt('../data/_final/val_tgt.csv', delimiter=','))
test_src = torch.from_numpy(genfromtxt('../data/_final/test_src.csv', delimiter=','))
test_tgt = torch.from_numpy(genfromtxt('../data/_final/test_tgt.csv', delimiter=','))

train = torch.utils.data.TensorDataset(train_src, train_tgt)
val = torch.utils.data.TensorDataset(val_src, val_tgt)
test = torch.utils.data.TensorDataset(test_src, test_tgt)
train_loader = torch.utils.data.DataLoader(train, batch_size=50, shuffle=True)

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    seq_len = 105
    nbatch = data.target_tensor.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data.data_tensor = data.data_tensor.narrow(0, 0, nbatch * batch_size)
    data.target_tensor = data.target_tensor.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data.data_tensor = data.data_tensor.unsqueeze(0).view(-1, batch_size, seq_len).contiguous()
    data.target_tensor = data.target_tensor.view(batch_size, -1).contiguous().unsqueeze(0)
    #print('data.data_tensor.view', data.data_tensor)
    #print('data.target_tensor.view', data.target_tensor)
    return data

eval_batch_size = 10
train_data = batchify(train, batch_size)
#val_data = batchify(val, eval_batch_size)
#test_data = batchify(test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

model = m.Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())

###############################################################################
# Training the Model
###############################################################################

batch_per_epoch = 1
for epoch in range(10):
    losses = []
    for batch_count in range(batch_per_epoch):
        model.zero_grad()

        print('train_data.data_tensor', train_data.data_tensor)
        print('train_data.target_tensor', train_data.target_tensor)
        #print('type(train_data.data_tensor)[batch_count]', type(train_data.data_tensor[batch_count]))

        preds = model(autograd.Variable(train_data.data_tensor.type(torch.LongTensor)[batch_count]))

        loss = criterion(preds.view(-1, model.vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if (batch_count % 20 == 0):
            print('Loss: ', losses[-1])
