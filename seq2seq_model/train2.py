import os
import argparse
import logging

import torch
from torch import autograd, nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from numpy import genfromtxt
import numpy as np
from torch.autograd import Variable

import data
import model as m

import torchtext
from torchtext import data, datasets
from evalTest import eval,test
#from torchtext.vocab import GloVe
from vecHandler import Vecs

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment', help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')

# learning
parser.add_argument('-mf', type=int, default=1, help='min_freq for vocab [default: 1]') #
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]') #
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]') #
parser.add_argument('-opt', type=str, default='adamax', help='optimizer [default: adamax]') #

# model
parser.add_argument('-net-type', type=str, default='gru', help='network type [default: gru]')
parser.add_argument('-num-layers', type=int, default=4, help='number of layers [default: 1]') #
parser.add_argument('-hidden-sz', type=int, default=300, help='hidden size [default: 300]') #
parser.add_argument('-num-dir', type=int, default=2, help='number of directions [default: 2]') #
parser.add_argument('-emb-dim', type=int, default=300, help='number of embedding dimension [default: 300]') #
parser.add_argument('-embfix', type=str, default=False, help='fix the embeddings [default: False]') #
parser.add_argument('-pretr-emb', type=str, default=False, help='use pretrained embeddings') #
parser.add_argument('-dropout', type=float, default=.5, help='dropout rate [default: .5]')

# options
parser.add_argument('-save-path', type=str, default='./saved_models', help='path to save models [default: ./saved_models]')
parser.add_argument('-folder', type=str, default='', help='folder to save models [default: '']')
parser.add_argument('-acc-thresh', type=float, default=40, help='top1 accuracy threshold to save model')
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
if args.mf > 1: args.save_path_full += '_mf' + str(args.mf)

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

################################################################################
# Prepare Dataset
################################################################################

TEXT = SourceField()
LABEL = TargetField()

train, val, test = data.TabularDataset.splits(
    # ms_draw data
    path='../ms_draw/', train='draw-train.tsv',
    validation='draw-dev.tsv', test='draw-test.tsv', format='tsv',
    fields=[('src', TEXT), ('tgt', LABEL)])



max_len = 50
# filter_pred

TEXT.build_vocab(train, max_size=50000)
LABEL.build_vocab(train, max_size=50000)
input_vocab = TEXT.vocab
output_vocab = LABEL.vocab

# NOTE: If the source field name and the target field name
# are different from 'src' and 'tgt' respectively, they have
# to be set explicitly before any training or inference
# seq2seq.src_field_name = 'src'
# seq2seq.tgt_field_name = 'tgt'

# Prepare loss
weight = torch.ones(len(TEXT.vocab))
pad = LABEL.vocab.stoi[LABEL.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()


seq2seq = None
optimizer = None

# Initialize model
hidden_size=128
bidirectional = True

encoder = EncoderRNN(
                vocab_size=len(TEXT.vocab),
                max_len=max_len,
                hidden_size=args.hidden_sz,
                input_dropout_p=0,
                dropout_p=args.dropout,
                n_layers=args.num_layers,
                bidirectional= args.num_dir==2,
                rnn_cell=args.net_type,
                variable_lengths=False
                )

decoder = DecoderRNN(
                vocab_size=len(LABEL.vocab),
                max_len=max_len,
                hidden_size=args.hidden_sz * 2 if bidirectional else 1,
                sos_id=LABEL.sos_id, # Add to params
                eos_id=LABEL.eos_id, # Add to params
                n_layers=args.num_layers,
                rnn_cell=args.net_type,
                bidirectional= args.num_dir==2,
                input_dropout_p=0,
                dropout_p=args.dropout,
                use_attention=False
                )

seq2seq = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
    seq2seq.cuda()

for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)

# Optimizer and learning rate scheduler can be customized by
# explicitly constructing the objects and pass to the trainer.
#
optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
scheduler = StepLR(optimizer.optimizer, 1)
optimizer.set_scheduler(scheduler)

# train
t = SupervisedTrainer(loss=loss, batch_size=32,
                      checkpoint_every=50,
                      print_every=10, expt_dir=opt.expt_dir)

seq2seq = t.train(seq2seq, train,
                  num_epochs=6, dev_data=val,
                  optimizer=optimizer,
                  teacher_forcing_ratio=0.5,
                  resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)
