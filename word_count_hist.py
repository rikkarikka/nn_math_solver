from collections import Counter
import string
text = open('./data/train.can.red.eq.id', 'r')
text_output = ''
for line in text:
    text_output = text_output + line.strip() + ' '
stripped_text = ''.join(ch for ch in text_output if ch not in set(string.punctuation))
text.close()
histogram = Counter(stripped_text.split(' '))
max_word_len = max(len(x) for x in histogram.keys())
for i in histogram.items():
    print (i[0] + (max_word_len - len(i[0])) * ' ', i[1], i[1] * '*')
total = str(sum(c[1] for c in histogram.items()))
print ('TOTAL' + (max_word_len - len('TOTAL')) * ' ', total, 'words')
