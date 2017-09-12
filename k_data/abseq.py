import sys
import json
import os
from collections import defaultdict, OrderedDict

def toFloat(x):
  x = x.replace(",","")
  try:
    return float(x)
  except:
    return False


path = sys.argv[1]
with open('keq.txt') as f:
  equations = f.read().split("\n")
#if True:

abseq = []
absurf = []
for f in os.listdir(path):
  if f[0] == '.': continue
  print(f)
  with open(path+'/'+f) as g:
    data = json.load(g)
    
  math_eq = equations[int(f.split(".")[0])] 

  #anon math problem
  for x in ")(*+=-/;":
    math_eq = math_eq.replace(x," "+x+" ")
  math_nums = {toFloat(x):None for x in math_eq.split() if toFloat(x)}
  # propers & pronouns
  
  script = [] 
  surface = []
  nerd = defaultdict(list)
  nerctr = 0
  numctr = 0
  for sentidx,s in enumerate(data['sentences']):
    sentnum = sentidx + 1
    txt = []
    lastner = None
    for x in s['tokens']:
      if x['ner']!="O":
        if x['ner'] != lastner:
          nerctr+=1
          lastner = x['ner']
        if toFloat(x['word']):
          #t = str(toFloat(x['word']))+"|"+x['ner']+"_"+str(nerctr)
          t = "NUMBER_"+str(numctr)
          numctr+=1
          math_nums[toFloat(x['word'])] = t
        elif x['ner'] == "PERSON":
          t = x['ner']+"_"+str(nerctr)
        else:
          t = x['word']
          #t = x['word']+"|"+x['ner']+"_"+str(nerctr)
      else:
        if lastner:
          lastner = None
        t = x['word']
      txt.append(t)
    surf = " ".join(txt)
    for x in math_nums:
      if str(int(x)) in surf:
        print('replacing: ',str(x))
        t = "NUMBER_"+str(numctr)
        surf = surf.replace(str(int(x)),t)
        math_nums[x] = t
        numctr+=1
    surface.append(surf)

    #make script
    deps = [x for x in s["collapsed-ccprocessed-dependencies"] if x['dep'] in ['nsubj','dobj','iobj']]
    di = OrderedDict()
    for d in deps:
       if d['governor'] not in di:
         di[d['governor']] = OrderedDict()
       di[d['governor']][d['dep']] = d['dependent']
    depst = []
    sentscr = []
    for k in di:
      st = "pred:" + txt[int(k)-1]
      for v in di[k]:
        if v == "nsubj":
          st = "nsubj:"+txt[int(di[k][v])-1] + " " + st
        elif v == "dobj":
          st += " dobj:" + txt[int(di[k][v])-1]
        elif v == "iobj":
          st += " iobj:" + txt[int(di[k][v])-1]
      sentscr.append(st)
    script.append(" ; ".join(sentscr))
    if txt[-1] in "?.!":
      script.append(txt[-1])
    else:
      script.append(".")

  #fix eq
  vard = {}
  varnum = 0
  finaleq = []
  for x in math_eq.split():
    if x[0].isalpha():
      if x not in vard:
        vard[x] = "VAR_"+str(varnum)
        varnum+=1
      finaleq.append(vard[x])
    elif toFloat(x) in math_nums:
      if math_nums[toFloat(x)]:
        finaleq.append(math_nums[toFloat(x)])
      else:
        finaleq.append(x)
    else:
      finaleq.append(x)

  #with open("abs_eqs/"+f[:-5],'w') as g:
  #  g.write(" ".join(finaleq)+"\n")
  abseq.append(" ".join(finaleq))
  absurf.append(" ".join(surface))

  #with open("surfaces/"+f[:-5],'w') as g:
  #  g.write(" ".join(surface))
with open("abseq.txt",'w') as f:
  f.write("\n".join(abseq))
with open("absurf.txt",'w') as f:
  f.write("\n".join(absurf))
