import sys
from re import split

with open("kabseq.can.txt") as f:
  data = f.read().strip().split("\n")

newdata = []
for eq in data:
  eqs = split("[\+\*\=\-\(\)\/]",eq)
  eqs = [x for x in eqs if "_" in x]
  eq2 = []
  for x in eqs:
    if x not in eq2: eq2.append(x)

  eqs = eq2
  num = "A"
  var = "A"
  for x in eqs:
    if "NUM" in x:
      repl = x.split("_")[0]+"_"+num
      num = chr(ord(num)+1)
    else:
      repl = x.split("_")[0]+"_"+var
      var = chr(ord(var)+1)
    eq = eq.replace(x,repl)
  print(eq)
