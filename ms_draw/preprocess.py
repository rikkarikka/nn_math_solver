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
            output.write(d['sQuestion'] + '\t')
            result = ''
            for eq in d['lEquations']:
                 result += str(eq) + ' ; '
            result = result[:-3]
            output.write(result + '\n')

if __name__ == '__main__':
    main()
