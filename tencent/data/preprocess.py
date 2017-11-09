import json
import numpy as np
import random
import math
import re
import sys

def main():
    split('./Math23K.json', './Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt')
    json2txt('./Math23K-train.txt','./Math23K.json',   './src-train.txt',  './tgt-train.txt')
    json2txt('./Math23K-dev.txt','./Math23K.json',     './src-val.txt',    './tgt-val.txt')
    json2txt('./Math23K-test.txt','./Math23K.json',    './src-test.txt',   './tgt-test.txt')

def split(json_path, train_path, dev_path, test_path):
    data = json.loads(open(json_path).read())
    random.shuffle(data)

    # remove uncommon equations
    print('Started with', len(data), 'examples')
    for d in data:
        d['segmented_text'], d['equation'] = preprocess(d['segmented_text'], d['equation'])
    data = mostCommon(data, 80)
    print('Filtered down to', len(data), 'examples')


    # 5 fold cross validation
    k_test = 5 # fold to use for test

    k = 5
    fold_size = math.floor(np.shape(data)[0] / k)

    for i in range(1,6):
        output = open('fold' + str(i) + '.txt', 'w')
        for d in data[(i-1) * fold_size: i * fold_size]:
            output.write(d['id'] + '\n')
        output.close()

    train_dev = []
    for i in range(1,6):
        if not i == k_test:
            train_dev = np.append(train_dev, open('fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_dev)
    test = open('fold' + str(k_test) + '.txt').readlines()

    # Train
    output = open(train_path, 'w')
    for d in train_dev[0:-500]:
        output.write(d + '\n')
    output.close()

    # Dev
    output = open(dev_path, 'w')
    for d in train_dev[-500:]:
        output.write(d + '\n')
    output.close()

    # Test
    output = open(test_path, 'w')
    for d in test:
        output.write(d + '\n')
    output.close()

def mostCommon(data, percent):
    # returns PERCENT of data by # of equation occurences

    equation, count= np.unique([d['equation'] for d in data], return_counts=True)
    indices = np.asarray((equation, count)).T[:,1].astype(int).argsort()
    result = np.asarray([[equation[i], count[i]] for i in indices])

    total_eqs = np.sum(np.asarray(result[:,1]).astype(int))
    num_equations_removed = 0
    occurences = 1
    while num_equations_removed < total_eqs * (1 - percent / 100):
        print('Removing equations with', occurences, 'occurences...')
        equations_to_remove = result[:,0][np.asarray(result[:,1]).astype(int) == occurences]
        for eq in equations_to_remove:
            eq = eq.strip()
            for d in data:
                if d['equation'].strip() == eq:
                    data.remove(d)
                    num_equations_removed += 1
        print('total # equations removed:', num_equations_removed)
        occurences += 1
    return data


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
    for token in question:
        if isFloat(token):
            for symbol in equation:
                if symbol == token:
                    equation[equation.index(symbol)] = '[' + chr(97 + i) + ']'
            for q in question:
                if q == token:
                    question[question.index(q)] = '[' + chr(97 + i) + ']'
            i += 1

    question = ' '.join(question) + '\n'
    equation = ' '.join(equation) + '\n'
    return question, equation

def json2txt(indices_path, json_path, output_path_src, output_path_tgt):
    json_indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
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

if __name__ == '__main__':
    main()
