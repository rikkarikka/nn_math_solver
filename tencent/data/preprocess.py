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

    # 5 fold cross validation
    k_test = 5 # fold to use for test
    
    k = 5
    fold_size = math.floor(np.shape(data)[0] / k)
    print(fold_size)
    for i in range(1,6):
        output = open('fold' + str(i) + '.txt', 'w')
        for d in data[(i-1) * fold_size: i * fold_size]:
            output.write(d['id'] + '\n')
        output.close()

    # Train
    output = open(train_path, 'w')
    for d in data[500:-500]:
        output.write(d['id'] + '\n')
    output.close()


    # Dev
    output = open(dev_path, 'w')

    for d in data[:500]:
        output.write(d['id'] + '\n')
    output.close()

    # Test
    output = open(test_path, 'w')
    for d in data[-500:]:
        output.write(d['id'] + '\n')
    output.close()

def json2txt(indices_path, json_path, output_path_src, output_path_tgt):
    json_indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
    output_src = open(output_path_src, 'w')
    output_tgt = open(output_path_tgt, 'w')
    for d in data:
        if int(d['id']) in json_indices:

            #handle fractions and % and numbers with units
            question = d['segmented_text'].replace('%', ' % ')
            equation = d['equation']

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

            tokens = np.array(question.split())
            i = 0

            for token in tokens:
                if isFloat(token):
                    for symbol in equation:
                        if symbol == token:
                            equation[equation.index(symbol)] = '[' + chr(97 + i) + ']'
                    token = '[' + chr(97 + i) + ']'
                    i += 1
                output_src.write(token + ' ')
            output_src.write('\n')


            # Preprocess Equations
            """
            for eq in d['Template']:
                symbols = eq.split()
                for i,symbol in enumerate(symbols):
                    if symbol not in ['+', '-', '*', '/', '(', ')', '='] and not isFloat(symbol):
                        symbols[i] = '[' + symbol + ']'
                for symbol in symbols:
                    result += str(symbol) + ' '
                result += ' ; '
            result = result[:-3]
            """
            output_tgt.write(' '.join(equation) + '\n')


def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    main()
