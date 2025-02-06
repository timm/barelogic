#!/usr/bin/env python3 -B
# <!--- vi: set ts=2 sw=2 sts=2 et : --->
"""   
nb.py : Naive Bayes   
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License   
   
OPTIONS:   
   -k k      low frequency Bayes hack   = 1   
   -m m      low frequency Bayes hack   = 2   
   -p p      distance formula exponent  = 2   
   -r rseed  random number seed         = 1234567891   
   -t train  training csv file          = data/misc/auto93.csv   
"""
import re,ast,sys,math,random
rand = random.random
Big  = 1E32


#----------------------------------------------------------------------------------------
def adds(src, i=None):
  for x in src:
    i = i or (Num() if isinstance(x[0],(int,float)) else Sym())
    add(x,i)
  return i

def ent(d):
  N = sum(n for n in d.values())
  return -sum(n/N * math.log(n/N.2) for n in d.values())

def coerce(s):
  try: return ast.literal_eval(s)
  except Exception: return s

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      if arg == "-"+k[0]:
        d[k] = coerce("False" if str(v) == "True"  else (
                      "True"  if str(v) == "False" else (
                      sys.argv[c+1] if c < len(sys.argv) - 1 else str(v))))

def show(x):
  it = type(x)
  if it == float: return str(round(x,the.decs))
  if it == list:  return ', '.join([show(v) for v in x])
  if it == dict:  return "("+' '.join([f":{k} {show(v)}" for k,v in x.items()])+")"
  if it == o:     return x.__class__.__name__ + show(x.__dict__)
  if it == str:   return '"'+str(x)+'"'
  if callable(x): return x.__name__
  return str(x)

#----------------------------------------------------------------------------------------
def eg__the(_):
  "show settings"
  print(the)

def eg_h(_):
  "show help"
  print(__doc__)
  [print(f"   {re.sub('^eg','',k).replace('_','-'):9s} {fun.__doc__}")
   for k,fun in globals().items() if k[:3] == "eg_"]

#----------------------------------------------------------------------------------------
the= o(**{m[1]:coerce(m[2]) for m in re.finditer(r"-\w+\s*(\w+).*=\s*(\S+)",__doc__)})

if __name__ == "__main__":
  cli(the.__dict__)
  for i,s in enumerate(sys.argv):
    if fun := vars().get("eg" + s.replace("-","_")):
      arg = None if i==len(sys.argv) - 1 else sys.argv[i+1]
      random.seed(the.rseed)
      fun(coerce(arg))
