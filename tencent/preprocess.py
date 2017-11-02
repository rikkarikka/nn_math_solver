import json
import numpy as np
import random
import math

def main():
    #split('./Math23K.json', './Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt')
    jsonToTsv('./Math23K-train.txt','./Math23K.json',   './Math23K-train-src.tsv',  './Math23K-train-tgt.tsv')
    jsonToTsv('./Math23K-dev.txt','./Math23K.json',     './Math23K-dev-src.tsv',    './Math23K-dev-tgt.tsv')
    jsonToTsv('./Math23K-test.txt','./Math23K.json',    './Math23K-test-src.tsv',   './Math23K-test-tgt.tsv')

def split(json_path, train_path, dev_path, test_path):
    data = json.loads(open(json_path).read())
    random.shuffle(data)
    split_pts = [0, math.floor(len(data)*.8), math.floor(len(data)*.9), len(data)]

    # Train
    output = open(train_path, 'w')
    for d in data[split_pts[0]:split_pts[1]]:
        output.write(d['id'] + '\n')
    output.close()


    # Dev
    output = open(dev_path, 'w')
    for d in data[split_pts[1]+1:split_pts[2]]:
        output.write(d['id'] + '\n')
    output.close()

    # Test
    output = open(test_path, 'w')
    for d in data[split_pts[2]+1:split_pts[3]]:
        output.write(d['id'] + '\n')
    output.close()



def jsonToTsv(indices_path, json_path, output_path_src, output_path_tgt):
    json_indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
    output_src = open(output_path_src, 'w')
    output_tgt = open(output_path_tgt, 'w')
    for d in data:
        if int(d['id']) in json_indices:

            '+', '-', '*', '/', '(', ')', '='
            equation = d['equation']
            equation = equation.replace('+', ' + ')
            equation = equation.replace('-', ' - ')
            equation = equation.replace('*', ' * ')
            equation = equation.replace('/', ' / ')
            equation = equation.replace('(', ' ( ')
            equation = equation.replace(')', ' ) ')
            equation = equation.replace('=', ' = ')
            equation = equation.split()

            # Preprocess Question
            tokens = np.array(d['segmented_text'].split())
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
