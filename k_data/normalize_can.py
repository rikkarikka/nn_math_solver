import sys
from re import split

with open(sys.argv[1]) as f:
  data = f.read().strip().split("\n")

newdata = []
for eq in data:
  if ";" in eq:
    a,b = eq.split(";")
    eq = a+" ; "+b if len(a)<len(b) else b+" ; "+a
  eqs = split("[\+\*\=\-\(\)\/;]",eq)
  eqs = [x for x in eqs if "_" in x]
  eq2 = []
  for x in eqs:
    if x not in eq2: eq2.append(x)

  eqs = eq2
  num = "A"
  var = "A"
  for x in eqs:
    if "NUM" in x:
      repl = x.split("_")[0]+"_"+num+" "
      num = chr(ord(num)+1)
    else:
      repl = x.split("_")[0]+"_"+var+" "
      var = chr(ord(var)+1)
    eq = eq.replace(x,repl)
  print(eq.strip())
