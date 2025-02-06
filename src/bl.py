#!/usr/bin/env python3 -B
# vi: set ts=2 sw=2 sts=2 et :
"""
nb.py : Naive Bayes
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License

OPTIONS:
   -d decs   decimap places for printing = 3
   -f file   training csv file           = ../../moot/optimize/misc/auto93.csv
   -k k      low frequency Bayes hack    = 1
   -m m      low frequency Bayes hack    = 2
   -p p      distance formula exponent   = 2
   -r rseed  random number seed          = 1234567891
"""
import re,ast,sys,math,random
rand = random.random
any  = random.choice
Big  = 1E32

class o:
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = lambda i    : show(i)

#----------------------------------------------------------------------------------------
def Num(txt=" ", at=0): 
  return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-Big, lo=Big,
           goal = 0 if txt[-1]=="-" else 0)

def Sym(txt=" ", at=0): 
  return o(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)

def Data(src): 
  return adds(src, o(it=Data, n=0, rows=[], cols=None))

def Cols(names):
  x,y,all = [], [],[]
  for col in [(Num if s[0].isupper() else Sym)(s,n) for n,s in enumerate(names)]:
    all.append(col)
    if col.txt[-1] != "X":
      (y if col.txt[-1] in "+-!" else x).append(col)
      if col.txt[-1] == "!": klass=col
  return o(it=Cols, names=names, all=all, x=x, y=y)

#----------------------------------------------------------------------------------------
def add(v,i):
  def _data():
    if i.cols: i.rows += [ [add( v[col.at], col) for col in i.cols.all] ]
    else: i.cols = Cols(v)
  def _sym():
   n = i.has[v] = 1 + i.has.get(v,0)
   if n > i.most: i.most, i.mode = n, v
  def _num():
    i.lo  = min(v, i.lo)
    i.hi  = max(v, i.hi)
    d     = v - i.mu
    i.mu += d / i.n
    i.m2 += d * (v -  i.mu)
    i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5
  # ----------
  if v != "?":
    i.n += 1 
    _sym()  if i.it is Sym else ( _num()  if i.it is Num else _data())
  return i 

def norm(v,col):
  print(1,v)
  print(2,col)
  return v if (v=="?" or col.it is Sym) else (v - col.lo) /  (col.hi - col.lo + 1/Big)

def mid(col): 
  return col.mu if col.it is Num else col.mode

def spread(col): 
  return col.sd if col.it is Num else ent(col.has)

def ydist(row, data):
  return (sum(abs(norm(row[col.at], col) - col.goal)**the.p for col in data.cols.y)  
          / len(data.cols.y)) ** (1/the.goal)

def eg__data(_):
  d=Data(csv(the.file))
  d.rows.sort(key=lambda row: ydist(row,d))
  [print(row) for row in d.rows[::50]]
  #showd({col.txt : mid(col) for col in d.cols.all})
  #showd({col.txt : spread(col) for col in d.cols.all})

#----------------------------------------------------------------------------------------
def like(lst, data, nall=100, nh=2):
  def _like1(v,col):
    if col.it is Sym: 
      return (sym.has.get(v,0) + the.m*prior) / (sym.n + the.m)
    else:
      sd    = col.sd + 1/Big
      nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
      denom = (2*math.pi*sd*sd) ** 0.5
      return max(0, min(1, nom/denom))
  #-----------------------------------
  prior = (data.n + the.k) / (nall + the.k*nh)
  likes = [like1(lst[col.at], col) for col in data.cols.x if lst[col.at]!="?"]
  return sum(math.log(like) for like in likes + [prior] if like>0)

def activeLearning(data):
  bests, todos = data.rows[:8],  random.shuffle(rows[8:])
  rests, todos = todos[:32], todos[32:]
  best,  rest  = adds(bests,Data()), adds(rests,Data())
  maybe = []
  for _ in range(the.actives):
    n = best.n + rest.n
    j = random.randint(0,len(todos))
    row = todo[j]
    maybe += [(likes(row,best,n, 2) / likes(row,rest,n,2), row,j)]
  _,_,j = max(maybe, key = lambda lrowj: lrowj[0])
  done += [todo.pop(j)]
  
#----------------------------------------------------------------------------------------
def adds(src, i=None):
  for x in src:
    i = i or (Num() if isinstance(x[0],(int,float)) else Sym())
    add(x,i)
  return i

def ent(d):
  N = sum(n for n in d.values())
  return -sum(n/N * math.log(n/N,2) for n in d.values())

def coerce(s):
  try: return ast.literal_eval(s)
  except Exception: return s

def csv(file):
  file = sys.stdin if file=="-" else open(file)
  with file as src:
    for line in src:
      line = re.sub(r'([\n\t\r ]|#.*)', '', line)
      if line: yield [coerce(s.strip()) for s in line.split(",")]

def eg__csv(_):
  [print(row) for row in list(csv(the.file))[::50]]

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      if arg == "-"+k[0]:
        d[k] = coerce("False" if str(v) == "True"  else (
                      "True"  if str(v) == "False" else (
                      sys.argv[c+1] if c < len(sys.argv) - 1 else str(v))))

def showd(x): print(show(x)); return x

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
