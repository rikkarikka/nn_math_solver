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

    text2ids = {}
    with open(opt.ids) as f:
        ids = f.readlines()
        for line in ids:
            text2ids.update({line.split()[0]:line.split()[1]})

    idstxt = open("IDs.txt", 'w')
    with open(opt.input_txt) as f:
            input_txt = f.readlines()
            for line in input_txt:
                if opt.src == 1:
                    words = line.split()
                    words2ids = ''
                    for w in words:
                        if text2ids.has_key(w):
                            words2ids = words2ids + text2ids.get(w) + ' '
                        else:
                            words2ids = words2ids + '<unkn>  '
                    idstxt.write(words2ids + '\n')
	        elif opt.src == 0:
                    idstxt.write('src == 0,')

def main():
    print('Running...')
    replaceWithIDs()
    print('Done...')

if __name__ == "__main__":
    main()
