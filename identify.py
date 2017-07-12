import argparse

parser = argparse.ArgumentParser(description='identify.py')

# **Preprocess Options**
parser.add_argument('-input_txt', required=True,
                    help="Path to the input data")
parser.add_argument('-ids', required=True,
                    help="Path to the text ID data")
parser.add_argument('-src',    type=int, default=1,
                    help="src data or tgt data")

opt = parser.parse_args()

def replaceWithIDs():
    "Generate text file with words replaced with IDs"

    idstxt = open("val.txt.id", 'w')
    with open(opt.input_txt) as f:
        input_txt = f.readlines()
        if opt.src == 1:
            text2ids = {}
            with open(opt.ids) as g:
                ids = g.readlines()
                for line_ids in ids:
                    text2ids.update({line_ids.split()[0]:line_ids.split()[1]})
            #print(text2ids)
            for line in input_txt:
                words = line.split()
                words2ids = ''
                for w in words:
                    if w.lower() in text2ids:
                        words2ids = words2ids + text2ids.get(w.lower()) + ' '
                    else:
                        words2ids = words2ids + '<unk> '
                idstxt.write(words2ids + '\n')
        elif opt.src == 0:
            text2ids = {}
            with open(opt.ids) as g:
                ids = g.readlines()
                for line_ids in ids:
                    key = line_ids.split(' |||  ')[0].lower() + '\n'
                    #idstxt.write('key: ' + key + '\n')
                    value = line_ids.split(' |||  ')[1].lower()
                    #idstxt.write('value: ' + value + '\n')
                    text2ids.update({key:value})
            #print(text2ids)
            for line in input_txt:
                #print('line: ' + line)
                #print('line.lower(): ' + line.lower())
                if line.lower() in text2ids:
                    idstxt.write(text2ids.get(line.lower()))
                else:
                    idstxt.write('<unk>' + '\n')


def main():
    print('Running...')
    replaceWithIDs()
    print('Done...')

if __name__ == "__main__":
    main()
