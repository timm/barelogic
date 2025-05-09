import fileinput
BIG=1E32

class o: 
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = lambda i: i.__class__.__name__ + str(i.__dict__)
  
the = o(p=2, file="../../moot/optimize/misc/auto93.csv")

class Col(o):
  def sub(i,x,n=1): return i.add(x,n,-1)
  def add(i,x,n=1,flip=1):
  	if x != "?":
  	  i.n += flip*n
  	  return i.add1(x,n,flip)
  	return x

class Sym(Col):
  def __init__(i,txt=" ",at=1): i.txt,i.at,i.n,i.has = txt, at, 0, {}
  def add1(i,x,n,flip): 
    i.has[x] = flip*n + (i.has[x] if x in i.has else 0) 
  
class Num(Col):
  def __init__(i,txt=" ",at=1): 
    i.txt,i.at,i.n,i.mu,i.m2 = txt, at, 0, 0, 0
    i.lo, i.hi, i.goal = BIG, -BIG, 0 if txt[-1]=="-" else 1
  def add1(i,x,n,flip):
    i.lo = min(x, i.lo)
    i.hi = max(x, i.hi)
    if  flip != 1 and i.n < 2:
      i.n = i.mu = i.sd = 0
    else:
      d = x - i.mu
      i.mu += flip * d / i.n
      i.m2 += flip * d * (x - i.mu) 

class Data(o):
  def __init__(i,src=[]): 
    i.rows, i.cols = [], None
    [i.add(row) for row in i.rows] 
  def add(i, row):
  	if not i.cols: i.cols = Cols(x)
    elif: i.rows.update(i.cols.add(row)) 
  def sub(i, row, purge=True):
  	if purge: i.rows.remove(x)
  	[i.sub(row[c.at]) for c in i.cols.all]	
  	
class Cols(o):
  def __init__(i,names):
    i.x, i.y, i.names = [],[],names
    i.all = [(Num if s[0].isupper() else Sym)(s,j) for j,s in enumerate(names)]
    for col in i.all: 
      if col.txt[-1] != "X": 
        (i.y if col.txt[-1] in "!+-" else i.x).append(col)
  def add(i,row): [c.add(row[c.at]) for c in i.all]
  def sub(i,row): [c.sub(row[c.at]) for c in i.all]
  	      
def adds(lst,i=None): 
  for x in lst:
  	i = i or (Sym if type(x) is str else Num)()
  	i.add(x)
  return i

def csv(file=None):
  buf = ""
  for line in fileinput.input(file):
    if line := line.split("#")[0].replace(" ", "").strip():
      buf += line
      if buf and buf[-1] != ",": 
        yield [coerce(x) for x in buf.split(",")]
        buf = ""
         
def coerce(x, specials= {'True':1, 'False':0, 'None':None}):
  try: return int(x)
  except:
    try: return float(x)
    except:
      x = x.strip()
      return specials[x] if x in specials else x