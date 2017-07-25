import argparse

parser = argparse.ArgumentParser(description='identify.py')

# **Preprocess Options**
#parser.add_argument('-src',    type=int, default=1,
                    #help="src data or tgt data")

#opt = parser.parse_args()

# **Global Variables**

# Input Files
train_src = './data/train.txt'
train_tgt = './data/train.eq'
val_src = './data/val.txt'
val_tgt = './data/val.eq'
test_src = './data/test.txt'
test_tgt = './data/test.eq'

# Dictionary Files
src_dict = './data/ids.atok.low.src.dict'
tgt_dict = './data/ids.atok.low.tgt.dict'

# Output Files
train_src_ids = './data/train.txt.id'
train_tgt_ids = './data/train.eq.id'
val_src_ids = './data/val.txt.id'
val_tgt_ids = './data/val.eq.id'
test_src_ids = './data/test.txt.id'
test_tgt_ids = './data/test.eq.id'

def replaceWithIDs(input_txt, dictionary, output_txt, src):
    "Generate text file with words replaced with IDs"

    idstxt = open(output_txt, 'w')
    with open(input_txt) as f:
        input_txt = f.readlines()
        if src == 1:
            text2ids = {}
            with open(dictionary) as g:
                ids = g.readlines()
                for line_ids in ids:
                    text2ids.update({line_ids.split()[0]:line_ids.split()[1]})
            for line in input_txt:
                words = line.split()
                words2ids = ''
                for w in words:
                    if w.lower() in text2ids:
                        words2ids = words2ids + text2ids.get(w.lower()) + ' '
                    else:
                        words2ids = words2ids + '<unk> '
                idstxt.write(words2ids + '\n')
        elif src == 0:
            text2ids = {}
            with open(dictionary) as g:
                ids = g.readlines()
                for line_ids in ids:
                    key = line_ids.split(' |||  ')[0].lower() + '\n'
                    value = line_ids.split(' |||  ')[1].lower()
                    text2ids.update({key:value})
            for line in input_txt:
                if line.lower() in text2ids:

                    idstxt.write(text2ids.get(line.lower()))
                else:
                    idstxt.write('<unk>' + '\n')


def main():
    print('Running identify.py...')
    replaceWithIDs(train_src, src_dict, train_src_ids, 1)
    replaceWithIDs(train_tgt, tgt_dict, train_tgt_ids, 0)
    replaceWithIDs(val_src, src_dict, val_src_ids, 1)
    replaceWithIDs(val_tgt, tgt_dict, val_tgt_ids, 0)
    replaceWithIDs(test_src, src_dict, test_src_ids, 1)
    replaceWithIDs(test_tgt, tgt_dict, test_tgt_ids, 0)
    print('identify.py complete...')

if __name__ == "__main__":
    main()
