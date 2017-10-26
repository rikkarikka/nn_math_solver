import json
import numpy as np

def main():
    jsonToTsv('./draw-train.txt','./draw.json', './draw-train.tsv')
    jsonToTsv('./draw-dev.txt','./draw.json', './draw-dev.tsv')
    jsonToTsv('./draw-test.txt','./draw.json', './draw-test.tsv')

def jsonToTsv(indices_path, json_path, output_path):
    indices = np.genfromtxt(indices_path).astype(int)
    data = json.loads(open(json_path).read())
    output = open(output_path, 'w')
    for d in data:
        if d['iIndex'] in indices:

            # Preprocess Question
            words = d['sQuestion'].split()
            for i,word in enumerate(words):
                for a in d['Alignment']:
                    value = None
                    if a['Value'] == int(a['Value']):
                        value = str(int(a['Value']))
                    else:
                        value = str(a['Value'])
                    if word == value:
                        words[i] = str('[' + str(a['coeff']) + ']') + ' '
                    #print(str(a['Value']), str(a['coeff']))
            if d['iIndex'] == 493569: print('words:', words)
            for word in words:
                output.write(word + ' ')
            output.write('\t')

            # Preprocess Equations
            result = ''
            for eq in d['Template']:
                symbols = eq.split()
                if d['iIndex'] == 493569: print('sybmols:', symbols)
                for i,symbol in enumerate(symbols):
                    if d['iIndex'] == 493569: print('sybmol:', symbol)
                    if d['iIndex'] == 493569: print(str(symbol not in ['+', '-', '*', '/', '(', ')', '=']) + '\n')
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
