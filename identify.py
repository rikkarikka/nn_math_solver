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
    with open(opt.input_txt) as f:
        input_txt = f.readlines()
    with open(opt.ids) as f:
        ids = f.readlines()

    dat = list(range(len(input_txt)))

    idstxt = open("IDs.txt", 'w')
    for i in dat[:]:
        idstxt.write(input_txt[i])

def main():
    print('Running...')
    replaceWithIDs()
    print('Done...')

if __name__ == "__main__":
    main()
