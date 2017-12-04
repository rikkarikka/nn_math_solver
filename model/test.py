import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable

import data
import model as m
from torchtext import data, datasets
import mydatasets
from evalTest import eval,test
from torchtext.vocab import GloVe
from vecHandler import Vecs

def main():
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    parser.add_argument('-path', type=str, default='', help='path to data file [default:]')
    args = parser.parse_args()

    #train_dev = open(train_dev_path).readlines()

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train_dev = data.TabularDataset(
        path='../tencent/data/train_dev_0.2.tsv',
        format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(train_dev)
    LABELS.build_vocab(train_dev)



    train_dev_iter = data.BucketIterator(train_dev, 8)
    model = torch.load('../tencent/models/common_0.2/net-lstm_e100_bs8_opt-adam_ly1_hs300_dr2_ed200_fembFalse_ptembFalse_drp0.3/acc94.00_e19.pt')

    (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(train_dev_iter, model, TEXT, args.emb_dim)
    print('ACCURACY:', accuracy)

if __name__ == '__main__':
    main()
