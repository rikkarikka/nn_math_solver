import json
import numpy as np
import random
import math

def main():
    #split('./Math23K.json', './Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt')
    jsonToTsv('./Math23K-train.txt','./Math23K.json', './Math23K-train.tsv')
    jsonToTsv('./Math23K-dev.txt','./Math23K.json', './Math23K-dev.tsv')
    jsonToTsv('./Math23K-test.txt','./Math23K.json', './Math23K-test.tsv')

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



def jsonToTsv(indices_path, json_path, output_path):
    json_indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
    output = open(output_path, 'w')
    for d in data:
        #print(d['iIndex'] in indices)
        if d['iIndex'] in json_indices:
            print(d['sQuestion'])

            # Preprocess Question
            tokens = np.array(d['sQuestion'].split())
            for a in d['Alignment']:
                indices = np.array([])
                indices = np.append(indices, np.where(tokens == '.')) # add . indicies
                indices = np.append(indices, np.where(tokens == '?')) ## add ? indicies
                indices += 1
                indices = np.append(indices, [0])
                indices.sort()
                tokens[int(indices[a['SentenceId']] + a['TokenId'])] = '[' + a['coeff'] + ']'
            for token in tokens:
                output.write(token + ' ')
            output.write('\t')

            # Preprocess Equations
            result = ''
            for eq in d['Template']:
                symbols = eq.split()
                for i,symbol in enumerate(symbols):
                    if symbol not in ['+', '-', '*', '/', '(', ')', '='] and not isFloat(symbol):
                        symbols[i] = '[' + symbol + ']'
                for symbol in symbols:
                    result += str(symbol) + ' '
                result += ' ; '
            result = result[:-3]
            output.write(result + '\n')

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    main()
