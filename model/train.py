import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
import time
import csv

import data
import model as m
from torchtext import data, datasets
import mydatasets

###############################################################################
# Training Parameters
###############################################################################

input_size = 3151
hidden_size = 3
num_classes = 839
batch_size = 11
learning_rate = .001
epochs = 10

###############################################################################
# Load data
###############################################################################

train_src = torch.from_numpy(genfromtxt('../data/_final/train_src.csv', dtype="i8", delimiter=',')).type(torch.LongTensor).t()+1
train_tgt = torch.from_numpy(genfromtxt('../data/_final/train_tgt.csv', dtype="i8", delimiter=',')).type(torch.LongTensor)
val_src = torch.from_numpy(genfromtxt('../data/_final/val_src.csv', dtype="i8", delimiter=',')).type(torch.LongTensor).t()+1
val_tgt = torch.from_numpy(genfromtxt('../data/_final/val_tgt.csv', dtype="i8", delimiter=',')).type(torch.LongTensor)
test_src = torch.from_numpy(genfromtxt('../data/_final/test_src.csv', dtype="i8", delimiter=',')).type(torch.LongTensor).t()+1
test_tgt = torch.from_numpy(genfromtxt('../data/_final/test_tgt.csv', dtype="i8", delimiter=',')).type(torch.LongTensor)

# Load Data using torchtext
TEXT = data.Field(use_vocab=False)
LABEL = data.Field(use_vocab=False, sequential=False)

train, val, test = mydatasets.MWP.splits(text_field=TEXT, label_field=LABEL,
                                        train_src=train_src, train_tgt= train_tgt,
                                        val_src=val_src, val_tgt=val_tgt,
                                        test_src=test_src, test_tgt=test_tgt)

# Make iterator for splits
print('Making interator for splits')
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_size=batch_size, device=-1)

###############################################################################
# Build the model
###############################################################################

model = m.Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())

###############################################################################
# Training the Model
###############################################################################

model.train()
for epoch in range(10):
    losses = []
    for batch_count,batch in enumerate(train_iter):
        model.zero_grad()
        inp = batch.text.t()
        preds = model(inp)
        loss = criterion(preds, batch.label)
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if (batch_count % 20 == 0):
            print('Loss: ', losses[-1])
