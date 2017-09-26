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

def main():
    args = parseParams()
    if int(torch.cuda.is_available()) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if not os.path.isdir(args.save_path_full):
        train(args)
    else:
        print('Previously Trained')

def train(args):
    ###############################################################################
    # Load data
    ###############################################################################
    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train, val, test = data.TabularDataset.splits(
        path='../new_data/kdata', train='_train.tsv',
        validation='_dev.tsv', test='_test.tsv', format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    prevecs = None
    if (args.pretr_emb == True):
        #print('Making vocab w/ glove.6B.' + str(args.emb_dim) + ' dim vectors')
        TEXT.build_vocab(train,vectors=GloVe(name='6B', dim=args.emb_dim))#wv_type="glove.6B")
        prevecs=TEXT.vocab.vectors
    else:
        TEXT.build_vocab(train)
    LABELS.build_vocab(train)
    #print('Making interator for splits...')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(args.batch_size, 256, 256),
        sort_key=lambda x: len(x.text), device=cuda)

    num_classes = len(LABELS.vocab)
    input_size = len(TEXT.vocab)
    ###############################################################################
    # Build the model
    ###############################################################################

    model = m.Model(input_size=input_size, hidden_size=args.hidden_sz,
                    num_classes=num_classes,prevecs=prevecs,
                    num_layers=args.num_layers, num_dir=args.num_dir,
                    batch_size=args.batch_size, emb_dim=args.emb_dim,
                    embfix=args.embfix, dropout=args.dropout,
                    net_type=args.net_type)
    criterion = nn.CrossEntropyLoss()
    # Select optimizer
    if (args.opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters())#, lr=args.lr)
    elif (args.opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters())#, lr=args.lr)
    elif (args.opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.5)#,lr=args.lr,momentum=0.5)
    else:
        #print('Optimizer unknown, defaulting to adamax')
        optimizer = torch.optim.Adamax(model.parameters())


    ###############################################################################
    # Training the Model
    ###############################################################################
    if cuda == 0:
        model = model.cuda()

    highest_t1_acc = 0
    highest_t1_acc_metrics = ''
    highest_t1_acc_params = ''
    results = ''
    for epoch in range(args.epochs):
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
        #print('Average loss over epoch ' + str(epoch) + ': ' + str(tot_loss/len(losses)))
        (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(val_iter, model, args.device)
        if accuracy > args.acc_thresh:
            save_path = '{}/acc{:.2f}_e{}.pt'.format(args.save_path_full, accuracy, epoch)
            if not os.path.isdir(args.save_path_full):
                os.makedirs(args.save_path_full)
            torch.save(model, save_path)

        if highest_t1_acc < accuracy:
            highest_t1_acc = accuracy
            highest_t1_acc_metrics = ('acc: {:6.4f}%({:3d}/{}) EPOCH{:2d} - loss: {:.4f} t5_acc: {:6.4f}%({:3d}' \
                    '/{}) MRR: {:.6f}'.format(accuracy, corrects, size,epoch, tot_loss/len(losses), t5_acc, t5_corrects, size, mrr))

            highest_t1_acc_params = (('PARAMETERS:' \
                    'net-%s' \
                    '_e%i' \
                    '_bs%i' \
                    '_opt-%s' \
                    '_ly%i' \
                    '_hs%i' \
                    '_dr%i'
                    '_ed%i' \
                    '_femb%s' \
                    '_ptemb%s' \
                    '_drp%.1f\n'
                    % (args.net_type, args.epochs, args.batch_size, args.opt, args.num_layers,
                    args.hidden_sz, args.num_dir, args.emb_dim, args.embfix, args.pretr_emb, args.dropout)))
        results += ('\nEPOCH{:2d} - loss: {:.4f}  acc: {:6.4f}%({:3d}/{}) t5_acc: {:6.4f}%({:3d}' \
                '/{}) MRR: {:.6f}'.format(epoch, tot_loss/len(losses), accuracy,
                                        corrects, size, t5_acc, t5_corrects, size,
                                        mrr))

    print(highest_t1_acc_metrics + '\n')
    writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params)

def writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params):
    if not os.path.isdir(args.save_path_full):
        os.makedirs(args.save_path_full)
    f = open(args.save_path_full + '/results.txt','w')
    f.write('PARAMETERS:\n' \
            'Net Type: %s\n' \
            #'Learning Rate: %f\n' \
            'Epochs: %i\n' \
            'Batch Size: %i\n' \
            'Optimizer: %s\n' \
            'Num Layers: %i\n' \
            'Hidden Size: %i\n' \
            'Num Directions: %i\n'
            'Embedding Dimension: %i\n' \
            'Fixed Embeddings: %s\n' \
            'Pretrained Embeddings: %s\n'
            'Dropout: %.1f\n'
            % (args.net_type, args.epochs, args.batch_size, args.opt, args.num_layers,
            args.hidden_sz, args.num_dir, args.emb_dim, args.embfix, args.pretr_emb, args.dropout))
    f.write(results)
    f.close()
    if highest_t1_acc > args.acc_thresh:
        g = open(args.save_path + args.folder+ '/best_models.txt','a')
        g.write(highest_t1_acc_metrics)
        g.write(highest_t1_acc_params)
        g.close()

def parseParams():
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]') #
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]') #
    parser.add_argument('-opt', type=str, default='adamax', help='optimizer [default: adamax]') #

    # model
    parser.add_argument('-net-type', type=str, default='lstm', help='network type [default: lstm]')
    parser.add_argument('-num-layers', type=int, default=1, help='number of layers [default: 1]') #
    parser.add_argument('-hidden-sz', type=int, default=100, help='hidden size [default: 300]') #
    parser.add_argument('-num-dir', type=int, default=2, help='number of directions [default: 2]') #
    parser.add_argument('-emb-dim', type=int, default=50, help='number of embedding dimension [default: 300]') #
    parser.add_argument('-embfix', type=str, default=False, help='fix the embeddings [default: False]') #
    parser.add_argument('-pretr-emb', type=str, default=False, help='use pretrained embeddings') #
    parser.add_argument('-dropout', type=float, default=.5, help='dropout rate [default: .5]')

    # options
    parser.add_argument('-save-path', type=str, default='./saved_models', help='path to save models [default: ./saved_models]')
    parser.add_argument('-folder', type=str, default='', help='folder to save models [default: '']')
    parser.add_argument('-acc-thresh', type=float, default=35, help='top1 accuracy threshold to save model')
    parser.add_argument('-device', type=int, default=1, help='GPU to use [default: 1]')
    args = parser.parse_args()

    args.embfix = (args.embfix == 'True')
    args.pretr_emb = (args.pretr_emb == 'True')

    args.save_path_full = args.save_path + \
                        args.folder + \
                        '/net-' + str(args.net_type) + \
                        '_e' + str(args.epochs) + \
                        '_bs' + str(args.batch_size) + \
                        '_opt-' + str(args.opt) + \
                        '_ly' + str(args.num_layers) + \
                        '_hs' + str(args.hidden_sz) + \
                        '_dr' + str(args.num_dir) + \
                        '_ed' + str(args.emb_dim) + \
                        '_femb' + str(args.embfix) + \
                        '_ptemb' + str(args.pretr_emb) + \
                        '_drp' + str(args.dropout)
    return args

if __name__ == '__main__':
    main()
