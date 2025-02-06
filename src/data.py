from lib import *


#----------------------------------------------------------------------------------------
def Num(txt=" ", at=0):
  return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-Big, lo=Big,
           goal = 0 if txt[-1]=="-" else 0)

def Sym(txt=" ", at=0):
  return o(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)

def Data(src):
  return adds(src, o(it=Data, n=0, rows={}, cols=None))

def Cols(names):
  x,y,all = [], [],[]
  for col in [(Num if s[0].isUpper(0) else Sym)(s,n) for n,s in enumerate(names)]:
    all.append(col)
    if col.txt[-1] != "X":
      (y if col.txt[-1] in "+-!" else x).append(col)
      if col.txt[-1] == "!": klass=col
  return o(it=Cols, names=names, all=all, x=x, y=y)

#----------------------------------------------------------------------------------------
def adds(src, i=None):
  for x in src:
    i = i or (Num() if isinstance(x[0],(int,float)) else Sym())
    add(x,i)
  return i

def add(v,i):
  """Add `v` to `i`."""
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

  if v != "?":
    i.n += 1
    _sym()  if i.it is Sym else ( _num()  if i.it is Num else _data())
  return i

def norm(v,col):
  return v if (v=="?" or col.it is Sym) else (v - col.lo) /  (col.hi - col.lo + 1/Big)

def mid(col): 
  return col.mu if col.it is Num else col.mode

def spread(col): 
  if col.it is Num: return col.sd
  N = sum(col.has.values())
  return -sum(n/N * math.log(n/N.2) for n in col.has.values())


