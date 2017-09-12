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
from evalTest import eval,test

###############################################################################
# Training Parameters
###############################################################################

hidden_size = 300
batch_size = 11
learning_rate = .001
epochs = 10
cuda = int(torch.cuda.is_available())-1
print("CUDA: ",cuda)

###############################################################################
# Load data
###############################################################################

TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
LABELS = data.Field(sequential=False)

train, val, test = data.TabularDataset.splits(
    path='../new_data/kdata', train='_train.tsv',
    validation='_dev.tsv', test='_test.tsv', format='tsv',
    fields=[('text', TEXT), ('label', LABELS)])

print("Making vocab w/ glove.6B.300 dim vectors")
TEXT.build_vocab(train,wv_type="glove.6B")
LABELS.build_vocab(train)
print('Making interator for splits...')
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(batch_size, 256, 256),
    sort_key=lambda x: len(x.text), device=cuda)

num_classes = len(LABELS.vocab)
input_size = len(TEXT.vocab)
###############################################################################
# Build the model
###############################################################################

model = m.Model(input_size=input_size, hidden_size=hidden_size,
                num_classes=num_classes,prevecs=TEXT.vocab.vectors)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())
#params = model.parameters()
#optimizer = torch.optim.SGD(params,lr=0.1)


###############################################################################
# Training the Model
###############################################################################
#model.train()
for epoch in range(epochs):
    losses = []
    train_iter.repeat=False
    for batch_count,batch in enumerate(train_iter):
        model.zero_grad()
        inp = batch.text.t()
        print("INP: ",inp.size())
        preds = model(inp)
        #print("PREDS: ",preds.size())
        #print("LABELS: ",batch.label.size())
        loss = criterion(preds, batch.label)
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if (batch_count % 20 == 0):
            print('Batch:', batch_count,', Loss: ', losses[-1].data)
    eval(val_iter, model)

#print('test', '2',TEXT,LABEL)
