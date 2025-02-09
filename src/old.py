#!/usr/bin/env python3 -B
"""
nb.py : Naive Bayes  
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  
  
OPTIONS:  

      -a acg    xploit or xplore or adapt   = xploit  
      -d decs   decimap places for printing = 3  
      -f file   training csv file           = ../../moot/optimize/misc/auto93.csv  
      -g guess  size of guess               = 0.5  
      -G Guesses max number of guesses      = 100  
      -k k      low frequency Bayes hack    = 1  
      -m m      low frequency Bayes hack    = 2  
      -p p      distance formula exponent   = 2  
      -r rseed  random number seed          = 1234567891  
      -s start  where to begin              = 4  
      -S Stop   where to end                = 32  
"""
import re,ast,sys,math,random
rand = random.random
one   = random.choice
BIG   = 1E32

class o:
   __init__ = lambda i,**d: i.__dict__.update(**d)
   __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)

#----------------------------------------------------------------------------------------
#      _  _|_  ._        _  _|_   _    
#     _>   |_  |   |_|  (_   |_  _>    

def Num(txt=" ", at=0):
   return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG,
                goal = 0 if txt[-1]=="-" else 1)

def Sym(txt=" ", at=0):
   return o(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)

def Cols(names):
   x,y,cols,klass = [], [], [], None
   for col in [(Num if s[0].isupper() else Sym)(s,n) for n,s in enumerate(names)]:
      cols.append(col)
      if col.txt[-1] != "X":
         (y if col.txt[-1] in "+-!" else x).append(col)
         if col.txt[-1] == "!": klass=col
   return o(it=Cols, names=names, all=cols, x=x, y=y, klass=klass)

def Data(src):
   return adds(src, o(it=Data, n=0, rows=[], cols=None))

def clone(data,src=[],rank=False):
   return adds(src.sort(key=lambda row: ydist(row,data)) if rank else src,
               Data([data.cols.names]))

#----------------------------------------------------------------------------------------
#          ._    _|   _.  _|_   _  
#     |_|  |_)  (_|  (_|   |_  (/_ 
#          |                       

# Return a summary of the items in `src`.    
# If `i` not given, infer it from the first items.
def adds(src, i=None):
   for x in src:
      out = i or (Num() if isinstance(x[0],(int,float)) else Sym())
      add(x,out)
   return out

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
      i.m2 += d * (v -   i.mu)
      i.sd  = 0 if i.n <2 else (i.m2/(i.n-1))**.5
   if v != "?":
      i.n += 1
      _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v

#----------------------------------------------------------------------------------------
#      _.        _   ._     
#     (_|  |_|  (/_  |   \/ 
#       |                /  

def norm(v,col):
   return v if (v=="?" or col.it is Sym) else (v - col.lo) /   (col.hi - col.lo + 1/BIG)

def mid(col): 
   return col.mu if col.it is Num else col.mode

def spread(col): 
   return col.sd if col.it is Num else ent(col.has)

def eg__data(_):
   d=Data(csv(the.file))
   print("klass ", d.cols.klass or "none")
   showd({col.txt : mid(col) for col in d.cols.all})
   showd({col.txt : spread(col) for col in d.cols.all})

#----------------------------------------------------------------------------------------
#       _|  o   _  _|_ 
#      (_|  |  _>   |_ 

def ydist(row, data):
   return (sum(abs(norm(row[col.at], col) - col.goal)**the.p for col in data.cols.y) 
          / len(data.cols.y)) ** (1/the.p)

def xdist(row1,row2,data):
   def _num(p,q,num):
      p, q = norm(p,num), norm(q,num)
      p = p if p !="?" else (1 if q<0.5 else 0)
      q = q if q !="?" else (1 if p<0.5 else 0)
      return abs(p-q)
   def _col(p,q,col):
      return 1 if p==q=="?" else (p != q if col.it is Sym else _num(p,q,col))
   return (sum(_col(row1[col.at], row2[col.at],col)**the.p for col in data.cols.x)
          / len(data.cols.x))**(1/the.p)

def eg_y(_):
   d=Data(csv(the.file))
   for j,row in enumerate(sorted(d.rows,key=lambda row: ydist(row,d))):
      if j<10 or j % 50 == 0: print(j,row,round(ydist(row,d),2))

#----------------------------------------------------------------------------------------
#      |_    _.       _    _ 
#      |_)  (_|  \/  (/_  _> 
#                /           

def like(lst, data, nall=100, nh=2):
   def _col(v,col): 
      if col.it is Sym: 
         return (col.has.get(v,0) + the.m*prior) / (col.n + the.m)
      sd    = col.sd + 1/BIG
      nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
      denom = (2*math.pi*sd*sd) ** 0.5
      return max(0, min(1, nom/denom))
   prior = (data.n + the.k) / (nall + the.k*nh)
   likes = [_col(lst[x.at], x) for x in data.cols.x if lst[x.at] != "?"]
   return sum(math.log(l) for l in likes + [prior] if l>0)

def acting(data):
   def _acquire(p, b,r):
      b,r = math.e**b, math.e**r
      q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
      return (b + r*q) / abs(b*q - r + 1/BIG) 

   def _guess(todo,done, cut):
      def _score(row):
         n = len(done)
         return _acquire(n/the.Stop, like(row,best,n,2), like(row,rest,n,2))
      best = clone(data, done[:cut])
      rest = clone(data, done[cut:])
      top,*others = sorted(todo[:the.Guesses], key=_score, reverse=True)
      return top, todo[the.Guesses:] + others

   _rank = lambda rows: clone(data,rows,rank=True).rows
   done  = _rank(data.rows[:the.start])
   todo  = shuffle(data.rows[the.start:])
   while len(todo) > 2  and len(done) < the.Stop :
      top,todo = _guess(todo, done, round(len(done) ** the.guess))
      done    += [top]
      done     = _rank(done)
   return done

#----------------------------------------------------------------------------------------
#      |  o  |_  
#      |  |  |_) 

def shuffle(lst): random.shuffle(lst); return lst

def ent(d):
   N = sum(n for n in d.values())
   return -sum(n/N * math.log(n/N,2) for n in d.values())

def coerce(s):
   try: return ast.literal_eval(s)
   except Exception: return s

def csv(file):
   with open(sys.stdin if file=="-" else file, encoding="utf-8") as src:
      for line in src:
         line = re.sub(r'([\n\t\r ]|#.*)', '', line)
         if line: yield [coerce(s.strip()) for s in line.split(",")]

def eg__csv(_):
   [print(row) for row in list(csv(the.file))[::50]]

# For command like flags that match the first letter of key, update that value. 
# For boolean values, flags need no arguments (we just negate the default)
def cli(d):
   for k,v in d.items():
      for c,arg in enumerate(sys.argv):
         if arg == "-"+k[0]:
            d[k] = coerce("False" if str(v) == "True"   else (
                          "True"   if str(v) == "False" else (
                          sys.argv[c+1] if c < len(sys.argv) - 1 else str(v))))

# Pretty print
def showd(x): print(show(x)); return x

# Convert `x` to a pretty string.
def show(x):
   it = type(x)
   if   it is str   : x = f'"{x}"'
   elif callable(x) : x = x.__name__ + '()'
   elif it is float : x = str(round(x,the.decs))
   elif it is list  : x = '['+', '.join([show(v) for v in x])+']'
   elif it is dict  : x = "("+' '.join([f":{k} {show(v)}" for k,v in x.items() 
                                                          if k[0] !="_"])+")"
   return str(x)

def eg__the(_): 
   "show settings"
   print(the)

def eg_h(_): 
   "show help"
   print(__doc__)
   [print(f"    {re.sub('^eg','',k).replace('_','-'):9s} {fun.__doc__}") 
    for k,fun in globals().items() if k[:3] == "eg_"]

#----------------------------------------------------------------------------------------
#      ._ _    _.  o  ._  
#      | | |  (_|  |  | | 

def main():
   cli(the.__dict__)
   for i,s in enumerate(sys.argv):
      if fun := globals().get("eg" + s.replace("-","_")):
         arg = None if i==len(sys.argv) - 1 else sys.argv[i+1]
         random.seed(the.rseed)
         fun(coerce(arg))

the= o(**{m[1]:coerce(m[2]) for m in re.finditer(r"-\w+\s*(\w+).*=\s*(\S+)",__doc__)})

if __name__ == "__main__":  main()
