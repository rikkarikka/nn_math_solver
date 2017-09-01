import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
import time

import data
import model as m
from torchtext import data, datasets
import mydatasets

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

train_src = torch.from_numpy(genfromtxt('../data/_final/train_src.csv', delimiter=',')).type(torch.LongTensor)
train_tgt = torch.from_numpy(genfromtxt('../data/_final/train_tgt.csv', delimiter=',')).type(torch.LongTensor)
val_src = torch.from_numpy(genfromtxt('../data/_final/val_src.csv', delimiter=',')).type(torch.LongTensor)
val_tgt = torch.from_numpy(genfromtxt('../data/_final/val_tgt.csv', delimiter=',')).type(torch.LongTensor)
test_src = torch.from_numpy(genfromtxt('../data/_final/test_src.csv', delimiter=',')).type(torch.LongTensor)
test_tgt = torch.from_numpy(genfromtxt('../data/_final/test_tgt.csv', delimiter=',')).type(torch.LongTensor)

# Load Data using torchtext
TEXT = data.Field(use_vocab=False)
LABEL = data.Field(use_vocab=False, sequential=False)

train, val, test = mydatasets.MWP.splits(text_field=TEXT, label_field=LABEL,
                                        train_src=train_src, train_tgt= train_tgt,
                                        val_src=val_src, val_tgt=val_tgt,
                                        test_src=test_src, test_tgt=test_tgt)

# print information about the data
print('train.fields', train.fields)
print('type(train)', type(train))
#print('train.examples[0].text', train.examples[0].text)
print('len(train)', len(train))
print('val.fields', val.fields)
print('len(val)', len(val))
print('test.fields', test.fields)
print('len(test)', len(test))
print('vars(train[0])', vars(train[0]))

print('Building training text vocab...')
#TEXT.build_vocab(val)
print('Building training labels vocab...')
#LABEL.build_vocab(val)

# make iterator for splits
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_size=batch_size)

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
batch_per_epoch = 1
for epoch in range(10):
    losses = []
    for batch_count in train_iter:
        model.zero_grad()
        preds = model(train_iter)

        loss = criterion(preds.view(-1, model.vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if (batch_count % 20 == 0):
            print('Loss: ', losses[-1])
"""
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    seq_len = 105
    nbatch = data.target_tensor.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data.data_tensor = data.data_tensor.narrow(0, 0, nbatch * batch_size)
    data.target_tensor = data.target_tensor.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data.data_tensor = data.data_tensor.unsqueeze(0).view(-1, batch_size, seq_len).contiguous().type(torch.LongTensor)
    data.target_tensor = data.target_tensor.view(batch_size, -1).contiguous().unsqueeze(0).type(torch.LongTensor)
    #print('data.data_tensor.view', data.data_tensor)
    #print('data.target_tensor.view', data.target_tensor)
    return data
"""
