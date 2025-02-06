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
from lib import *
from data import Num,Sym,Data, adds,add,norm,mid,spread

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      if arg == "-"+k[0]:
        d[k] = coerce("False" if str(v) == "True"  else (
                      "True"  if str(v) == "False" else (
                      sys.argv[c+1] if c < len(sys.argv) - 1 else str(v))))

the= o(**{m[1]:coerce(m[2]) for m in re.finditer(r"-\w+\s*(\w+).*=\s*(\S+)",__doc__)})

def eg__data(_):
  adds(csv(the.file), Data())


