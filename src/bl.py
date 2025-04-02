# vim: set sw=2:ts=2:et:
"""
nb.py : Naive Bayes    
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  
  
OPTIONS:  

      -a acq     xploit or xplore or adapt   = xploit  
      -d decs    decimal places for printing = 3  
      -f file    training csv file           = ../test/data/auto93.csv  
      -g guess   size of guess               = 0.5  
      -G Guesses max number of guesses       = 100  
      -k k       low frequency Bayes hack    = 1  
      -l leaf    min size of tree leaves     = 2
      -m m       low frequency Bayes hack    = 2  
      -p p       distance formula exponent   = 2  
      -r rseed   random number seed          = 1234567891  
      -s start   where to begin              = 4  
      -S Stop    where to end                = 32  
"""
import re,sys,math,random

rand  = random.random
shuffle=random.shuffle
one   = random.choice
some  = random.choices
BIG   = 1E32

#--------- --------- --------- --------- --------- --------- ------- -------
class o:
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)

def Num(txt=" ", at=0):
  return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG, 
           goal = 0 if txt[-1]=="-" else 1)

def Sym(txt=" ", at=0):
  return o(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)

def Cols(names):
  cols = o(it=Cols, x=[], y=[], all=[], klass=None)
  for n,s in enumerate(names):
    col = (Num if s[0].isupper() else Sym)(s,n)
    cols.all += [col]
    if col.txt[-1] != "X":
      (cols.y if col.txt[-1] in "+-!" else cols.x).append(col)
      if col.txt[-1] == "!": cols.klass=col
  return cols

def Data(src=[]): return adds(src, o(it=Data,n=0,rows=[],cols=None))

def clone(data, src=[]): return adds(src, Data([data.cols.names]))

#--------- --------- --------- --------- --------- --------- ------- -------
def adds(src, i=None):
  for x in src:
    if not i: return adds(src,Num() if isinstance(x,(int,float)) else Sym())
    add(x,i)
  return i

def add(v, i):
  def _data():
    if i.cols: i.rows  += [[add( v[col.at], c) for c in i.cols.all]]
    else     : i.cols   = Cols(v)
  def _sym():
    n = i.has[v] = 1 + i.has.get(v,0)
    if n > i.most: i.most, i.mode = n, v
  def _num():
    i.lo  = min(v, i.lo)
    i.hi  = max(v, i.hi)
    d     = v - i.mu
    i.mu += d / i.n
    i.m2 += d * (v -   i.mu)
    i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5

  if v != "?":
    i.n += 1
    _sym() if i.it is Sym else (_num() if i.it is Num else _data())
  return v

def sub(v, i):
   def _data(): [sub(v[col.at],col) for col in i.cols.all]  
   def _sym() : i.has[v] -= 1
   def _num():
     d     = v - i.mu
     i.mu -= d / i.n
     i.m2 -= d * (v - i.mu)
     i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5

   if v != "?":
     i.n -= 1
     _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v

#--------- --------- --------- --------- --------- --------- ------- -------
def norm(v, col):
   if v=="?" or col.it is Sym: return v
   return (v - col.lo) / (col.hi - col.lo + 1/BIG)

def mid(col): return col.mu if col.it is Num else col.mode

def spread(col): return col.sd if col.it is Num else ent(col.has)

#def delta(i,j): return abs(i.mu - j.mu) / ((i.sd**2/i.n + j.sd**2/j.n)**.5 + 1/BIG)

def ydist(row,  data):
  return (sum(abs(norm(row[c.at], c) - c.goal)**the.p for c in data.cols.y) 
          / len(data.cols.y)) ** (1/the.p)

def ydists(row, data): return sorted(rows, key=lambda row: ydist(row,data))

#--------- --------- --------- --------- --------- --------- ------- -------
def likes(lst, datas):
  n = sum(data.n for data in datas)
  return max(datas, key=lambda data: like(lst, data, n, len(datas)))

def like(lst, data, nall=100, nh=2):
  def _col(v,col): 
    if col.it is Sym: 
      return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
    sd    = col.sd + 1/BIG
    nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
    denom = (2*math.pi*sd*sd) ** 0.5
    return max(0, min(1, nom/denom))

  prior = (data.n + the.k) / (nall + the.k*nh)
  tmp   = [_col(lst[x.at], x) for x in data.cols.x if lst[x.at] != "?"]
  return sum(math.log(l) for l in tmp + [prior] if l>0)

#--------- --------- --------- --------- --------- --------- ------- -------
def acquire(p, b,r): 
  b,r = math.e**b, math.e**r
  q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
  return (b + r*q) / abs(b*q - r + 1/BIG) 

def activeLearn(data, guess=acquire):
  def _guess(row): 
     return guess(n/the.Stop, like(row,best,n,2), like(row,rest,n,2))

  n     =  the.start
  todo  =  shuffle(data.rows[n:])
  done  =  ydists(data.rows[:n], data)
  cut   =  round(n**the.guess)
  best  =  clone(data, done[:cut])
  rest  =  clone(data, done[cut:])
  while len(todo) > 2  and n < the.Stop:
    n += 1
    top, *others = sorted(todo[:the.Guesses], key=_guess, reverse=True)
    m = int(len(others)/2)
    todo = others[:m] + todo[the.Guesses:] + others[m:]
    add(top, best)
    best.rows = ydists(best.rows, data)
    if len(best.rows) > n**0.5:
      add( sub(best.rows.pop(-1), best), rest)
  return best.rows

#--------- --------- --------- --------- --------- --------- ------- -------
