#!/usr/bin/env python3 -B
"""
nb.py : Naive Bayes  
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  
  
OPTIONS:  

      -a acg    xploit or xplore or adapt   = xploit  
      -d decs   decimal places for printing = 3  
      -f file   training csv file           = ../test/data/auto93.csv  
      -g guess  size of guess               = 0.5  
      -G Guesses max number of guesses      = 100  
      -k k      low frequency Bayes hack    = 1  
      -m m      low frequency Bayes hack    = 2  
      -p p      distance formula exponent   = 2  
      -r rseed  random number seed          = 1234567891  
      -s start  where to begin              = 4  
      -S Stop   where to end                = 32  
"""
import re,sys,math,random
from typing import List, Dict, Generator
from typing import Any

rand  = random.random
one   = random.choice
some  = random.choices
BIG   = 1E32

# `Obj` allows for easy initialization and have a built-in pretty print.
class Obj:
   __init__ = lambda i,**d: i.__dict__.update(**d)
   __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)

number    = float  | int   #
atom      = number | bool | str # and sometimes "?"
row       = List[atom]
classes   = Dict[str,List[row]] # `str` is the class name

#----------------------------------------------------------------------------------------
#      _  _|_  ._        _  _|_   _    
#     _>   |_  |   |_|  (_   |_  _>    

# Define a numerical column with statistics.
def Num(txt: str = " ", at: int = 0) -> Obj:
   return Obj(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG,
             goal = 0 if txt[-1]=="-" else 1)

# Define a symbolic column with frequency counts.
def Sym(txt: str = " ", at: int = 0) -> Obj:
   return Obj(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)

# Define a collection of columns with metadata.
def Cols(names: List[str]) -> Obj:
   x,y,lst,klass = [], [], [], None
   for col in [(Num if s[0].isupper() else Sym)(s,n) for n,s in enumerate(names)]:
      lst.append(col)
      if col.txt[-1] != "X":
         (y if col.txt[-1] in "+-!" else x).append(col)
         if col.txt[-1] == "!": klass=col
   return Obj(it=Cols, names=names, all=lst, x=x, y=y, klass=klass)

# Define a dataset with rows and columns.
def Data(src: List[row], txt: str = "") -> Obj:
   return adds(src, Obj(it=Data, txt=txt or "", n=0, rows=[], cols=None))

# Return a dataset with the same structure as `data`. Optionally, rank rows.
def clone(data: Obj, src: List[row] = None, rank: bool = False) -> Obj:
   src = src or []
   return adds(src.sort(key=lambda row: ydist(row,data)) if rank else src,
               Data([data.cols.names]))

def Span(col, lo, hi=None):
   return Obj(it=Span, at=col.at, txt=col.txt, lo=lo, hi=hi or lo)

#----------------------------------------------------------------------------------------
#          ._    _|   _.  _|_   _  
#     |_|  |_)  (_|  (_|   |_  (/_ 
#          |                       

# Return a summary filled with `src`. If no summary supplied, assume one based on src[0].
def adds(src: List[Any], i: Any = None) -> Obj:
   for x in src:
      i = i or (Num() if isinstance(x,(int,float)) else Sym())
      add(x,i)
   return i

# Add a value to a struct (Num or Sym or Data).
def add(v: Any, i: Obj) -> Any:
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
      i.m2 += d * (v -   i.mu)
      i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5
   # Add ==>
   if v != "?":
      i.n += 1
      _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v

# Remove value from a struct.
def sub(v: Any, i: Obj) -> Any:
   def _data():
      [sub(v[col.at],col) for col in i.cols.all]  
   def _sym():
      i.has[v] -= 1
   def _num():
      d     = v - i.mu
      i.mu -= d / i.n
      i.m2 -= d * (v - i.mu)
      i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5
   # Sub ==>
   if v != "?":
      i.n -= 1
      _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v

#----------------------------------------------------------------------------------------
#      _.        _   ._     
#     (_|  |_|  (/_  |   \/ 
#       |                /  

# Normalize a value based on column statistics.
def norm(v: Any, col: Obj) -> Any:
   return v if (v=="?" or col.it is Sym) else (v - col.lo) /   (col.hi - col.lo + 1/BIG)

# Return the middle of a data.
def mids(data1: Obj) -> row:
  return [mid(col) for col in data1.cols.all]

# Return the central tendency of a column.
def mid(col: Obj) -> Any: 
   return col.mu if col.it is Num else col.mode

# Return the dispersion of a column.
def spread(col: Obj) -> float: 
   return col.sd if col.it is Num else ent(col.has)

# Report distance between two Nums, modulated in terms of the standard deviation."
def delta(i: Num, j: Num) -> float:
   return abs(i.mu - j.mu) / ((i.sd**2/i.n + j.sd**2/j.n)**.5 + 1/BIG)

#----------------------------------------------------------------------------------------
#       _|  o   _  _|_ 
#      (_|  |  _>   |_ 

# Compute the y-distance of a row from dataset.
def ydist(row1: row,  data: Obj) -> float:
   return (sum(abs(norm(row1[col.at], col) - col.goal)**the.p for col in data.cols.y) 
          / len(data.cols.y)) ** (1/the.p)

# Compute the x-distance between two rows.
def xdist(row1: row, row2: row, data: Obj) -> float:
   def _num(p,q,num):
      p, q = norm(p,num), norm(q,num)
      p = p if p !="?" else (1 if q<0.5 else 0)
      q = q if q !="?" else (1 if p<0.5 else 0)
      return abs(p-q)
   def _col(p,q,col):
      return 1 if p==q=="?" else (p != q if col.it is Sym else _num(p,q,col))
   # Xdist ==>
   return (sum(_col(row1[col.at], row2[col.at],col)**the.p for col in data.cols.x)
          / len(data.cols.x))**(1/the.p)

def neighbors(row1: row, rows: List[row], data: Obj) -> List[row]:
   return sorted(rows, key=lambda row2: xdist(row1,row2,data))

# def kcentroids(data,k=24,rows=None,samples=32):
#   rows = rows or data.rows
#   out  = [one(rows)]
#   for _ in range(1,k):
#     all,u = 0,[]
#     for _ = ranges(samples):
#       row = one(rows)
#       near = neighbors(row, out)[2]
#       all = all + push(u, {row=row, d=self:xdist(row,closest)^2}).d end 
#     local i,r = 1,all * math.random()
#     for j,x in pairs(u) do
#       r = r - x.d
#       if r <= 0 then i=j; break end end 
#     push(out, u[i].row)
#   end
#   return out end

#----------------------------------------------------------------------------------------
#      |_    _.       _    _ 
#      |_)  (_|  \/  (/_  _> 
#                /           

def likes(lst: List[Any], datas: List[Obj]) -> Obj:
   nall = sum(data.n for data in datas)
   return max(datas, key=lambda data: like(lst,data,nall,len(datas)))

# Compute the likelihood of a list belonging to a dataset.
def like(lst: List[Any], data: Obj, nall: int = 100, nh: int = 2) -> float:
   def _col(v,col): 
      if col.it is Sym: 
         return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
      sd    = col.sd + 1/BIG
      nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
      denom = (2*math.pi*sd*sd) ** 0.5
      return max(0, min(1, nom/denom))
   # Like ==>
   prior = (data.n + the.k) / (nall + the.k*nh)
   tmp = [_col(lst[x.at], x) for x in data.cols.x if lst[x.at] != "?"]
   return sum(math.log(l) for l in tmp + [prior] if l>0)

def acting(data : Obj):
   def _rank(rows): 
      return sorted(rows, key=lambda row1: ydist(row1,data))

   def _score(row1) : 
      return _want(n/the.Stop, like(row1,best,n,2), like(row1,rest,n,2))

   def _want(p, b,r): 
      b,r = math.e**b, math.e**r
      q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
      return (b + r*q) / abs(b*q - r + 1/BIG) 

   # Acting ==>
   n     = the.start
   todo  = shuffle(data.rows[n:])
   done  = _rank(data.rows[:n])
   cut   = round(n**the.guess)
   best  = clone(data, done[:cut])
   rest  = clone(data, done[cut:])
   while len(todo) > 2  and n < the.Stop :
      top,*others = sorted(todo[:the.Guesses], key=_score, reverse=True)
      todo = todo[the.Guesses:] + others
      add(top, best)
      n += 1
      best.rows = _rank(best.rows)
      if len(best.rows) > n**0.5:
         add( sub(best.rows.pop(-1), best), rest)
   return best.rows

def cuts(rows, data, Y=ydist, ything=Num):
   def _sym(sym,xys):
      ys= {}
      for x,y in xys:
          ys[x] = ys.get(x) or ything(txt=txt)
          add(y, ys[x])
      return (sum(ys.values(), key= lambda col1: col1.n*spred(col1)/len(xys)),
             [Span(col,txt) for col in ys.values()])

   def _num(num,xys):
      rhs1, rhs, lhs = {}, ything(), adds([y for _,y in xys], ything())
      for j,(_,y) in xys[::-1]: 
         add(y,rhs)
         rhs1[ len(xys) - j - 1 ] = (rhs.n * rhs.sd)/len(xys)
      lo,cut = BIG,None
      for j,(x,y) in enumerate(xys):
         add(y,lhs)
         if j < len(xy) - 1 and x != xys[j+1][0]:
           xpect = lhs.n/len(xys) * spread(lhs) + rhs1[j] 
           if xpect < lo:
              lo, cut = xpect,x
      return lo, ([Span(num,-BIG,cut),Span(num,cut,BIG)] if cut else [])

   def spans(col):
      xys = [(row[col.at], Y(row,data)) for row in rows if row[num.at] != "?"]
      yield (_sym if col.at is Sym else _num)(col, sorted(xys, key=first))

   return min(spans(col) for col in data.cols.x, key=first)[1]

def tree(data,Y=ydist, ything=Num, gaurd=None,):
   if len(data.rows) > the.leaf:
      here.kids=[]
      rows = data.rows
      for span in cuts(rows or data.rows, data,Y,ything):
         sub,rows = selects(rows,span)
         here.kids += [tree(clone(data,sub),Y,ything,guad=span)]
   return data 
 

def selects(rows, span): 
   yes,no = [],[]
   for row in rows: (yes if select(row,span) or no).append(row)
   return yes,no

def select(row,span):
   v= row[span.at
   return v=="?" or span.lo==span.hi==v or span.lo <= v < span.hi

#----------------------------------------------------------------------------------------
#      |  o  |_  
#      |  |  |_) 

# Return first item in a list
def first(lst:List) -> Any:
   return lst[0]

# Shuffle a list in place.
def shuffle(lst: List[Any]) -> List[Any]: random.shuffle(lst); return lst

# Compute entropy of a dictionary.
def ent(d: Dict[Any, int]) -> float:
   N = sum(n for n in d.values())
   return -sum(n/N * math.log(n/N,2) for n in d.values())


def coerce(s: str):
   try: return int(s)
   except Exception:
      try: return float(s)
      except Exception:
         s = s.strip()
         return True if s=="True" else (False if s=="False" else s)

# Parse a CSV file and yield rows.
def csv(file: str) -> Generator[List[Any], None, None]:
   with open(sys.stdin if file=="-" else file, encoding="utf-8") as src:
      for line in src:
         line = re.sub(r'([\n\t\r ]|#.*)', '', line)
         if line: yield [coerce(s) for s in line.split(",")]

# For command like flags that match the first letter of key, update that value. 
# For Boolean values, flags need no arguments (we just negate the default)
def cli(d: Dict[str, Any]) -> None:
   for k,v in d.items():
      for c,arg in enumerate(sys.argv):
         if arg == "-"+k[0]:
            d[k] = coerce("False" if str(v) == "True"   else (
                          "True"   if str(v) == "False" else (
                          sys.argv[c+1] if c < len(sys.argv) - 1 else str(v))))

# Pretty print
def showd(x: Any) -> Any: print(show(x)); return x

# Convert `x` to a pretty string. Round floats. For dicts, hide slots starting with "_".
def show(x: Any) -> str:
   it = type(x)
   if   it is str   : x = f'"{x}"'
   elif callable(x) : x = x.__name__ + '()'
   elif it is float : x = str(round(x,the.decs))
   elif it is list  : x = '['+', '.join([show(v) for v in x])+']'
   elif it is dict  : 
      x = "{"+' '.join([f":{k} {show(v)}" for k,v in x.items() if k[0] !="_"])+"}"
   return str(x)

def eg__the(_): print(the)
def eg_h(_): print(__doc__)

#----------------------------------------------------------------------------------------
#      ._ _    _.  o  ._  
#      | | |  (_|  |  | | 

# Main function to parse command-line arguments and run tests.
def main() -> None:
   cli(the.__dict__)
   for i,s in enumerate(sys.argv):
      if fun := globals().get("eg" + s.replace("-","_")):
         arg = None if i==len(sys.argv) - 1 else sys.argv[i+1]
         random.seed(the.rseed)
         fun(coerce(arg))

the= Obj(**{m[1]:coerce(m[2]) for m in re.finditer(r"-\w+\s*(\w+).*=\s*(\S+)",__doc__)})
random.seed(the.rseed)

if __name__ == "__main__":  main()
