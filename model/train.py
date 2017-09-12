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

shift = 2
input_size = 3150 + shift
hidden_size = 3
num_classes = 839 + shift
batch_size = 11
learning_rate = .001
epochs = 10

###############################################################################
# Load data
###############################################################################

train_src = torch.from_numpy(genfromtxt('../data/_final/train_src.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift
train_tgt = torch.from_numpy(genfromtxt('../data/_final/train_tgt.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift
val_src = torch.from_numpy(genfromtxt('../data/_final/val_src.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift
val_tgt = torch.from_numpy(genfromtxt('../data/_final/val_tgt.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift
test_src = torch.from_numpy(genfromtxt('../data/_final/test_src.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift
test_tgt = torch.from_numpy(genfromtxt('../data/_final/test_tgt.csv',
                    dtype="i8", delimiter=',')).type(torch.LongTensor)+shift

# Load Data using torchtext
TEXT = data.Field(use_vocab=False)
LABEL = data.Field(use_vocab=False, sequential=False)

train, val, test = mydatasets.MWP.splits(text_field=TEXT,label_field=LABEL,
                                        train_src=train_src,train_tgt=train_tgt,
                                        val_src=val_src,val_tgt=val_tgt,
                                        test_src=test_src,test_tgt=test_tgt)

# Make iterator for splits

print('Making interator for splits...')
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_size=batch_size, device=-1)

###############################################################################
# Build the model
###############################################################################

model = m.Model(input_size=input_size, hidden_size=hidden_size,
                num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())
#params = model.parameters()
#optimizer = torch.optim.SGD(params,lr=0.1)


###############################################################################
# Training the Model
###############################################################################
def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch_count,batch in enumerate(data_iter):
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        #if args.cuda:
        #    feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

def test(text, model, text_field, label_field):
    model.eval()
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return predicted.data[0][0]+1

#model.train()
for epoch in range(epochs):
    losses = []
    train_iter.repeat=False
    for batch_count,batch in enumerate(train_iter):
        model.zero_grad()
        inp = batch.text.t()
        preds = model(inp)
        loss = criterion(preds, batch.label)
        loss.backward()
        optimizer.step()
        losses.append(loss)

        if (batch_count % 20 == 0):
            print('Batch:', batch_count,', Loss: ', losses[-1].data)
    eval(val_iter, model)

#print('test', '2',TEXT,LABEL)
