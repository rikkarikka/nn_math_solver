import sys
from torchtext import data

TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
LABELS = data.Field()

train, val, test = data.TabularDataset.splits(
    path='kdata', train='_train.tsv',
    validation='_dev.tsv', test='_test.tsv', format='tsv',
    fields=[('text', TEXT), ('labels', LABELS)])

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(16, 256, 256),
    sort_key=lambda x: len(x.text), device=0)

TEXT.build_vocab(train,wv_type="glove.6B")
LABELS.build_vocab(train)
