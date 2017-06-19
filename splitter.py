import sys
from random import shuffle

with open("all_eqs.txt") as f:
  eq = f.readlines()

with open("all_text.txt") as f:
  txt = f.readlines()

dat = list(range(len(txt)))
shuffle(dat)

valeq = open("val.eq",'w')
valtxt = open("val.txt",'w')
for i in dat[:200]:
  valeq.write(eq[i])
  valtxt.write(txt[i])

valeq = open("test.eq",'w')
valtxt = open("test.txt",'w')
for i in dat[200:400]:
  valeq.write(eq[i])
  valtxt.write(txt[i])

valeq = open("train.eq",'w')
valtxt = open("train.txt",'w')
for i in dat[400:]:
  valeq.write(eq[i])
  valtxt.write(txt[i])
