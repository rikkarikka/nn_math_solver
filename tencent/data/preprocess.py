import json
import numpy as np
import random
import math
import re
import sys
import torch
from torchtext import data, datasets

sys.path.append('../../sni/model')
import model

def main():

    # LOAD DATA
    data = json.loads(open('./Math23K.json').read())

    # PREPROCESS DATA
    for d in data:
        d['segmented_text'], d['equation'] = preprocess(d['segmented_text'], d['equation'])

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    #print('Preforming cross validation splits...')
    #crossValidation(data, k = 5, k_test=5)

    # SAVE SPLIT INDICES
    split('./Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt', k_test=5)

    # SAVE SRC/TGT files
    train_indices = np.genfromtxt('./Math23K-train.txt').astype(int)
    dev_indices = np.genfromtxt('./Math23K-dev.txt').astype(int)
    test_indices = np.genfromtxt('./Math23K-test.txt').astype(int)
    json2txt(train_indices, data,   './src-train.txt',  './tgt-train.txt')
    json2txt(dev_indices,   data,   './src-val.txt',    './tgt-val.txt')
    json2txt(test_indices,  data,   './src-test.txt',   './tgt-test.txt')

    # REMOVE TEST FOLD BEFORE COUNTING UNCOMMON EQUATIONS
    data = [d for d in data if int(d['id']) not in test_indices]

    # REMOVE UNCOMMON EQUATIONS
    print('Removing uncommon equations...')
    print('Started with', len(data), 'examples')
    common_data, uncommon_data = mostCommon(data, .8)
    print('Filtered down to', len(common_data), 'examples')

    # SAVE SRC/TGT FILES (FILTERED DATA)
    train_dev_indices = np.append(train_indices, dev_indices)
    json2txt(train_dev_indices, common_data,    './src-train_dev_0.8_common.txt',   './tgt-train_dev_0.8_common.txt')
    json2txt(train_dev_indices, uncommon_data,  './src-train_dev_0.8_uncommon.txt', './tgt-train_dev_0.8_uncommon.txt')

    # SAVE TSV FILES
    txt2tsv('./src-train.txt',  './tgt-train.txt', './train.tsv')
    txt2tsv('./src-val.txt',  './tgt-val.txt', './val.tsv')
    txt2tsv('./src-test.txt',  './tgt-test.txt', './test.tsv')
    txt2tsv('./src-train_dev_0.8_common.txt',  './tgt-train_dev_0.8_common.txt', './train_dev_0.8_common.tsv')
    txt2tsv('./src-train_dev_0.8_uncommon.txt',  './tgt-train_dev_0.8_uncommon.txt', './train_dev_0.8_uncommon.tsv')

def crossValidation(data, k = 5, k_test=5):
    # Saves k folds
    # k: k fold cross validation
    # k_test: fold to use for test

    random.shuffle(data)
    fold_size = math.floor(np.shape(data)[0] / k)
    for i in range(1, k + 1):
        output = open('fold' + str(i) + '.txt', 'w')
        for d in data[(i-1) * fold_size: i * fold_size]:
            output.write(d['id'] + '\n')
        output.close()
        print('fold' + str(i) + '.txt' + ' saved')

def split(train_path, dev_path, test_path, k_test=5):
    train_dev = []
    for i in range(1,6):
        if not i == k_test:
            train_dev = np.append(train_dev, open('fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_dev)
    test = open('fold' + str(k_test) + '.txt').readlines()

    # Train
    output = open(train_path, 'w')
    for d in train_dev[0:-1000]:
        output.write(d)
    output.close()
    print(train_path + ' saved')

    # Dev
    output = open(dev_path, 'w')
    for d in train_dev[-1000:]:
        output.write(d)
    output.close()
    print(dev_path + ' saved')

    # Test
    output = open(test_path, 'w')
    for d in test:
        output.write(d)
    output.close()
    print(test_path + ' saved')

def mostCommon(data, percent):
    # returns PERCENT of data by # of equation occurences

    equation, count= np.unique([d['equation'] for d in data], return_counts=True)
    indices = np.asarray((equation, count)).T[:,1].astype(int).argsort()
    result = np.asarray([[equation[i], count[i]] for i in indices])
    removed = np.array([])

    total_eqs = np.sum(np.asarray(result[:,1]).astype(int))
    occurences = 1
    while len(removed) < total_eqs * (1 - percent):
        print('Removing equations with', occurences, 'occurences...')
        equations_to_remove = result[:,0][np.asarray(result[:,1]).astype(int) == occurences]
        for eq in equations_to_remove:
            eq = eq.strip()
            removed = np.append(removed, [d for d in data if d['equation'].strip() == eq])
            data = [d for d in data if not d['equation'].strip() == eq]

        print('total # equations removed:', len(removed))
        occurences += 1
    return data, removed


def preprocess(question, equation):
    #handle fractions and % and numbers with units
    question = question.replace('%', ' % ')

    fractions = re.findall('\(\d+\)/\(\d+\)', question)
    for i,fraction in enumerate(fractions):
        question = question.replace(fraction, str(sys.maxsize - i))
        equation = equation.replace(fraction, str(sys.maxsize - i))

    equation = equation.replace('+', ' + ')
    equation = equation.replace('-', ' - ')
    equation = equation.replace('*', ' * ')
    equation = equation.replace('/', ' / ')
    equation = equation.replace('(', ' ( ')
    equation = equation.replace(')', ' ) ')
    equation = equation.replace('=', ' = ')
    equation = equation.replace('^', ' ^ ')
    equation = equation.split()

    question = re.sub(r'(\d+)([A-z]{1,2})', r'\1 \2', question)

    # Preprocess Question

    question = question.split()

    i = 0

    question = ['null', 'null', 'null'] + question + ['null', 'null', 'null']
    question_copy = [t for t in question]

    model = torch.load('../../sni/models/sni_best_model.pt')
    model.eval()

    for j,token in enumerate(question):
        example = question_copy[j-3:j+4]
        if isFloat(token) and isSignificant(model, example):
            for symbol in equation:
                if symbol == token:
                    equation[equation.index(symbol)] = '[' + chr(97 + i) + ']'
            for q in question:
                if q == token:
                    question[question.index(q)] = '[' + chr(97 + i) + ']'
            i += 1

    question = question[3:-3]

    question = ' '.join(question) + '\n'
    equation = ' '.join(equation) + '\n'
    return question, equation

def json2txt(json_indices, data, output_path_src, output_path_tgt):
    output_src = open(output_path_src, 'w')
    output_tgt = open(output_path_tgt, 'w')
    for d in data:
        if int(d['id']) in json_indices:
            question, equation = preprocess(d['segmented_text'], d['equation'])
            output_src.write(question)
            output_tgt.write(equation)
    output_src.close()
    output_tgt.close()

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def isSignificant(model, example):
    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)
    fields=[('text', TEXT), ('label', LABELS)]


    print('isSignificant??')

    example = [example, '']
    example = data.Example.fromlist(example, fields)

    dataset = data.Dataset([example], fields)

    TEXT.build_vocab(dataset)
    LABELS.build_vocab(dataset)

    iterator = data.Iterator(dataset, batch_size=1)
    batch = data.batch(dataset, fields)

    output = model(batch)
    print(output)
    return(True)

def txt2tsv(src_path, tgt_path, tsv_path):
    src_txt = open(src_path).readlines()
    tgt_txt = open(tgt_path).readlines()
    tsv = open(tsv_path, 'w')
    for i in range(len(src_txt)):
        tsv.write(src_txt[i].strip() + '\t' + tgt_txt[i].strip() +'\n')

if __name__ == '__main__':
    main()
