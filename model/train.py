import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt

import data
import model as m
from torchtext import data, datasets
import mydatasets
from evalTest import eval,test
from torchtext.vocab import GloVe

###############################################################################
# Training Parameters
###############################################################################
parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-num-layers', type=int, default=1, help='number of layers [default: 1]')
parser.add_argument('-num-directions', type=int, default=2, help='number of directions [default: 2]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden-sz', type=int, default=300, help='hidden size [default: 300]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

# options
parser.add_argument('-save-path', type=str, default='./saved_models', help='path to save models [default: ./saved_models]')
parser.add_argument('-save-prefix', type=str, default='_default_save_prefix_', help='path to save models [default: ./saved_models]')
parser.add_argument('-snapshot', type=str, default='snapshot', help='filename of model snapshot [default: snapshot]')
args = parser.parse_args()

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
TEXT.build_vocab(train,vectors=GloVe(name='6B'))#wv_type="glove.6B")
LABELS.build_vocab(train)
print('Making interator for splits...')
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(args.batch_size, 256, 256),
    sort_key=lambda x: len(x.text), device=cuda)

num_classes = len(LABELS.vocab)
input_size = len(TEXT.vocab)
###############################################################################
# Build the model
###############################################################################

model = m.Model(input_size=input_size, hidden_size=args.hidden_sz,
                num_classes=num_classes,prevecs=TEXT.vocab.vectors)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())
#params = model.parameters()
#optimizer = torch.optim.SGD(params,lr=100,momentum=0.5)


###############################################################################
# Training the Model
###############################################################################
print('CUDA?', str(cuda == 0))
if cuda == 0:
    model = model.cuda()

print('Training Model...')
for epoch in range(args.epochs):
    print('Starting Epoch ' + str(epoch) + '...')
    losses = []
    tot_loss = 0
    train_iter.repeat=False
    for batch_count,batch in enumerate(train_iter):
        model.zero_grad()
        inp = batch.text.t()
        #print("INP: ",inp.size())
        preds = model(inp)
        #print("PREDS: ",preds.size())
        #print("LABELS: ",batch.label.size())
        loss = criterion(preds, batch.label)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        tot_loss += loss.data[0]

        #if (batch_count % 20 == 0):
            #print('Batch: ', batch_count, '\tLoss: ', str(losses[-1].data[0]))
    print('Average loss over epoch ' + str(epoch) + ': ' + str(tot_loss/len(losses)))
    eval(val_iter, model)

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    save_prefix = os.path.join(args.save_path, args.snapshot)
    save_path = '{}_epoch{}.pt'.format(save_prefix, epoch)
    torch.save(model, save_path)

#print('test', '2',TEXT,LABEL)
