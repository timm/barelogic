"""
bl.py : barelogic, XAI for active learning + multi-objective optimization
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  

OPTIONS:  

      -a acq        xploit or xplore or adapt   = xploit  
      -b bootstraps num of bootstrap samples    = 512
      -B BootConf   bootstrap threshold         = 0.95
      -c cliffConf  cliffs' delta threshold     = 0.197
      -C Cohen      Cohen threshold             = 0.35
      -d decs       decimal places for printing = 3  
      -f file       training csv file           = ../test/data/auto93.csv  
      -g guess      size of guess               = 0.5  
      -G Guesses    max number of guesses       = 100  
      -k k          low frequency Bayes hack    = 1  
      -K Kuts       max discretization zones    = 17
      -l leaf       min size of tree leaves     = 2
      -m m          low frequency Bayes hack    = 2  
      -p p          distance formula exponent   = 2  
      -r rseed      random number seed          = 1234567891  
      -s start      where to begin              = 4  
      -S Stop       where to end                = 32  
      -t tiny       min size of leaves of tree  = 4
"""
import re,sys,math,time,random

rand  = random.random
one   = random.choice
some  = random.choices
BIG   = 1E32

#--------- --------- --------- --------- --------- --------- ------- -------
class o:
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)

def Num(of=" ", at=0):
  return o(it=Num, of=of, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG, 
           rank=0, # used by the stats functions, ignored otherwise
           goal = 0 if str(of)[-1]=="-" else 1)

def Sym(of=" ", at=0):
  return o(it=Sym, of=of, at=at, n=0, has={})

def Cols(names):
  cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
  for n,s in enumerate(names):
    col = (Num if s[0].isupper() else Sym)(s,n)
    cols.all += [col]
    if s[-1] != "X":
      (cols.y if s[-1] in "+-!" else cols.x).append(col)
      if s[-1] == "!": cols.klass = col
  return cols

def Data(src=[]): return adds(src, o(it=Data,n=0,rows=[],cols=None))

def clone(data, src=[]): return adds(src, Data([data.cols.names]))

#--------- --------- --------- --------- --------- --------- ------- -------
def adds(src, i=None):
  for x in src:
    if not i: return adds(src,Num() if isNum(x) else Sym())
    add(x,i)
  return i

def add(v,i,  n=1): # n only used for fast sym add
  def _sym(): 
    i.has[v] = n  + i.has.get(v,0)
  def _data():
    if not i.cols: i.cols = Cols(v)  # called on first row
    else:  i.rows += [[add( v[col.at], col) for col in i.cols.all]]
  def _num():
    i.lo  = min(v, i.lo)
    i.hi  = max(v, i.hi)
    d     = v - i.mu
    i.mu += d / i.n
    i.m2 += d * (v -   i.mu)
    i.sd  = 0 if i.n <=2  else (max(0,i.m2)/(i.n-1))**.5

  if v != "?":
    i.n += n 
    _sym() if i.it is Sym else (_num() if i.it is Num else _data())
  return v

def sub(v,i,  n=1):
   def _data(): [sub(v[col.at],col) for col in i.cols.all]  
   def _sym() : 
     i.has[v] -= n
   def _num():
     if i.n < 2: i.mu = i.sd = 0
     else:
       d     = v - i.mu
       i.mu -= d / i.n
       i.m2 -= d * (v - i.mu)
       i.sd  = (max(0,i.m2)/(i.n-1))**.5

   if v != "?":
     i.n -= n
     _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v

#--------- --------- --------- --------- --------- --------- ------- -------
def norm(v, col):
   if v=="?" or col.it is Sym: return v
   return (v - col.lo) / (col.hi - col.lo + 1/BIG)

def mid(col): return col.mu if col.it is Num else max(i.has,key=i.has.get)

def spread(c): 
  if c.it is Num: return c.sd
  return -sum(n/c.n * math.log(n/c.n,2) for n in c.has.values() if n > 0)

def ydist(row,  data):
  return (sum(abs(norm(row[c.at], c) - c.goal)**the.p for c in data.cols.y) 
          / len(data.cols.y)) ** (1/the.p)

def ydists(rows, data): return sorted(rows, key=lambda row: ydist(row,data))

def yNums(rows,data): return adds(ydist(row,data) for row in rows)

#--------- --------- --------- --------- --------- --------- ------- -------
def likes(lst, datas):
  n = sum(data.n for data in datas)
  return max(datas, key=lambda data: like(lst, data, n, len(datas)))

def like(row, data, nall=100, nh=2):
  def _col(v,col): 
    if col.it is Sym: 
      return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
    sd    = col.sd + 1/BIG
    nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
    denom = (2*math.pi*sd*sd) ** 0.5
    return max(0, min(1, nom/denom))

  prior = (data.n + the.k) / (nall + the.k*nh)
  tmp   = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)

#--------- --------- --------- --------- --------- --------- ------- -------
def actLearn(data):
  def _guess(row): 
    return _acquire(n/the.Stop, like(row,best,n,2), like(row,rest,n,2))
  def _acquire(p, b,r): 
    b,r = math.e**b, math.e**r
    q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
    return (b + r*q) / abs(b*q - r + 1/BIG) 

  n     =  the.start
  todo  =  data.rows[n:]
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
def bestCurts
def cuts(rowss, col):
  def _cut(x):
    if col.it is Sym: return x
    if x==col.hi: x -= 1/BIG
    return col.lo + int(norm(x,col) * the.Kuts)*(col.hi - col.lo)

  n,tmp = 0,{}
  for y,rows in rowss.items():
    for row in rows:
      n = n + 1
      x = row[col.at]
      if x != "?":
        k = _cut(x)
        tmp[k] = tmp.get(k, None) or Sym(of=k, at=col.at)
        add(y, tmp[k])
  return col,*pastes(col,n, sorted(tmp.values(), key=lambda xy: xy.x.lo))
 
def pastes(col, n,cuts):
  def _add(sym1,sym2): [add(n,sym1,x) for x,n in sym2.has.values()]
  def _sub(sym1,sym2): [sub(n,sym1,x) for x,n in sym2.has.values()]

  def _sym(): return sum(cut.n/n * spread(cut) for cut in cuts), [
                     cut.of for cut in cuts]

  def _num():
    cut, xpect, left, right = None, BIG, Sym(), Sym()
    for sym in cuts: _add(right,sym) 
    for sym in cuts:
      _add(left, _sub(right, sym))
      tmp = (left.n*spread(left) + right.n*spread(right)) / n
      if tmp < xpect:
        xpect,cut = tmp, sym.of
    return xpect, [cut]

  return (_num if col.it is Num else _sym)()
  
#--------- --------- --------- --------- --------- --------- ------- -------
def tree(rows0,data):
   def yfun(row): return ydist(row,data)
   def ys(rows) return adds(yfun(row) for row in rows)
   def _spread(xys):
     return sum(xy.y.n/len(rows) * spread(xy.y) for xy in xys)

   def _grow(rows, yeps, lvl=0, guard=None):
     kids=[]
     if len(rows) > the.tiny and spread(ys(rows)) > yeps:
       for xy in min([discretize(rows, yfun, c) for c in datas[0].cols.x],
                     key=_spread)
          if rows1 := _grow(select(rows,xy.x), yeps, lvl+1, guard=xy):
             kids += [o(guard=guard, lvl=lvl, rows=rows1)]
       if kids: return kids

   return _grow(rows0, spread(adds(ys(rows0))) * the.Cohen)

#--------- --------- --------- --------- --------- --------- ------- -------
def delta(i,j): 
  return abs(i.mu - j.mu) / ((i.sd**2/i.n + j.sd**2/j.n)**.5 + 1/BIG)

# non-parametric significance test From Introduction to Bootstrap, 
# Efron and Tibshirani, 1993, chapter 20. https://doi.org/10.1201/9780429246593"""
def bootstrap(vals1, vals2):
    x,y,z = adds(vals1+vals2), adds(vals1), adds(vals2)
    yhat  = [y1 - mid(y) + mid(x) for y1 in vals1]
    zhat  = [z1 - mid(z) + mid(x) for z1 in vals2] 
    n     = 0
    for _ in range(the.bootstraps):
      n += delta(adds(some(yhat,k=len(yhat))), 
                 adds(some(zhat,k=len(zhat)))) > delta(y,z) 
    return n / the.bootstraps >= (1- the.BootConf)

# Non-parametric effect size. Threshold is border between small=.11 and medium=.28 
# from Table1 of  https://doi.org/10.3102/10769986025002101
def cliffs(vals1,vals2):
   n,lt,gt = 0,0,0
   for x in vals1:
     for y in vals2:
        n += 1
        if x > y: gt += 1
        if x < y: lt += 1 
   return abs(lt - gt)/n  < the.cliffConf # 0.197) 

def vals2RankedNums(d, eps=0, reverse=False):
  def _samples():            return [_sample(d[k],k) for k in d]
  def _sample(vals,of=" "): return o(vals=vals,num=adds(vals,Num(of=of)))
  def _same(b4,now):         return (abs(b4.num.mu - now.num.mu) < eps or
                                    cliffs(b4.vals, now.vals) and 
                                    bootstrap(b4.vals, now.vals))

  out,tmp = {},[]
  for now in sorted(_samples(), key=lambda z:z.num.mu, reverse=reverse):
    if tmp and _same(tmp[-1], now): 
      tmp[-1] = _sample(tmp[-1].vals + now.vals)
    else: 
      tmp += [ _sample(now.vals) ]
    now.num.meta= tmp[-1].num 
    now.num.meta.rank = chr( 97 + len(tmp) - 1 )
    out[now.num.of] = now.num
  return out

#--------- --------- --------- --------- --------- --------- ------- -------
def isNum(x): return isinstance(x,(float,int))

def coerce(s):
  try: return int(s)
  except Exception:
    try: return float(s)
    except Exception:
      s = s.strip()
      return True if s=="True" else (False if s=="False" else s)

def csv(file):
  with open(sys.stdin if file=="-" else file, encoding="utf-8") as src:
    for line in src:
      line = re.sub(r'([\n\t\r ]|#.*)', '', line)
      if line: yield [coerce(s) for s in line.split(",")]

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      if arg == "-"+k[0]:
        new = sys.argv[c+1] if c < len(sys.argv) - 1 else str(v)
        d[k] = coerce("False" if str(v) == "True"  else (
                      "True"  if str(v) == "False" else new))

def showd(x): print(show(x)); return x

def show(x):
  it = type(x)
  if   it is str   : x= f'"{x}"'
  elif callable(x) : x= x.__name__ + '()'
  elif it is float : x= str(round(x,the.decs))
  elif it is list  : x= '['+', '.join([show(v) for v in x])+']'
  elif it is dict  : x= "{"+' '.join([f":{k} {show(v)}" 
                                   for k,v in x.items() if str(k)[0] !="_"])+"}"
  elif it is XY    : x= show_xy(x)
  return str(x)

def main():
  cli(the.__dict__)
  for n,s in enumerate(sys.argv):
    if fun := globals().get("eg" + s.replace("-","_")):
      arg = "" if n==len(sys.argv) - 1 else sys.argv[n+1]
      random.seed(the.rseed)
      fun(coerce(arg))

#--------- --------- --------- --------- --------- --------- ------- -------
def eg__the(_): print(the)

def eg__cols(_): 
  s="Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+"
  [print(col) for col in Cols(s.split(",")).all]

def eg__csv(file): 
  rows =list(csv(file or the.file))
  assert 3192 == sum(len(row) for row in rows)
  for row in rows[1:]: assert type(row[0]) is int

def eg__data(file):
  data=Data(csv(file or the.file))
  assert 3184 == sum(len(row) for row in data.rows)
  for row in data.rows: assert type(row[0]) is int
  [print(col) for col in data.cols.all]
  nums = adds(ydist(row,data) for row in data.rows)
  print(o(mu=nums.mu, sd=nums.sd))

def dump(d): print(len(d.rows)); [print(col) for col in d.cols.all]

def eg__addSub(file):
  data=Data(csv(file or the.file))
  dump(data)
  cached=data.rows[:]
  while data.rows: sub(data.rows.pop(), data)
  dump(data)
  for row in cached: add(row,data)
  dump(data)
  for row in data.rows: assert -17 < like(row,data,1000,2) < -10

def eg__clone(file):
  data=Data(csv(file or the.file))
  dump(data)
  data2=clone(data,src=data.rows)
  dump(data2)

def eg__stats(_):
   def c(b): return 1 if b else 0
   G  = random.gauss
   R  = random.random
   n  = 32
   b4 = [G(10,1) for _ in range(n)]
   d  = 0
   while d < 2:
     now=[x+d*R() for x in b4]
     b1=cliffs(b4,now)
     b2=bootstrap(b4,now)
     showd(o(d=d,cliffs=c(b1), boot=c(b2), agree=c(b1==b2)))
     d+= 0.1

def eg__rank(_):
   G  = random.gauss
   n=100
   for k,num in vals2RankedNum(dict( asIs  = [G(10,1) for _ in range(n)],
                                copy1 = [G(20,1) for _ in range(n)],
                                now1  = [G(20,1) for _ in range(n)],
                                copy2 = [G(40,1) for _ in range(n)],
                                now2  = [G(40,1) for _ in range(n)],
                                ), the.Cohen).items():
      showd(o(r=num.meta.rank, mu=num.meta.mu, k=k))

def eg__actLearn(file,  repeats=20):
  file = file or the.file
  name = re.search(r'([^/]+)\.csv$', file).group(1)
  data = Data(csv(file))
  b4   = yNums(data.rows,data)
  now  = Num()
  t1   = time.perf_counter_ns()
  for _ in range(repeats):
    random.shuffle(data.rows)
    add(ydist(actLearn(data)[0],data), now)
  t2  = time.perf_counter_ns()
  print(o(win= (b4.mu - now.mu) /(b4.mu - b4.lo),
          rows=len(data.rows),x=len(data.cols.x),y=len(data.cols.y),
          lo0=b4.lo, mu0=b4.mu, hi0=b4.hi, mu1=now.mu,sd1=now.sd,
          ms = int((t2-t1)/repeats/10**6),
          stop=the.Stop,name=name))

def eg__fast(file):
  def rx1(data):
    random.shuffle(data.rows)
    return ydist( actLearn(data)[0], data)
  experiment1(file or the.file,
              repeats=20, 
              samples=[128,64,32,16,8],
              fun=rx1)

def eg__acts(file):
  def rx1(data):
    random.shuffle(data.rows)
    return [ydist(actLearn(data)[0], data)]
  experiment1(file or the.file,
              repeats=20, 
              samples=[256,128,64,32,16,8],
              fun=rx1)

def experiment1(file, repeats=20, samples=[32,16,8],
                      fun=lambda data1: ydist(actLearn(data)[0],data)):
  name = re.search(r'([^/]+)\.csv$', file).group(1)
  data = Data(csv(file))
  rx   = dict(b4 = [ydist(row,data) for row in data.rows])
  asIs = adds(rx["b4"])
  t1   = time.perf_counter_ns()
  for the.Stop in samples:
    rx[the.Stop] = []
    for _ in range(repeats): rx[the.Stop] +=  fun(data) 
  t2 = time.perf_counter_ns()
  report = dict(rows = len(data.rows), 
                lo   = f"{asIs.lo:.2f}",
                x    = len(data.cols.x), 
                y    = len(data.cols.y),
                ms   = round((t2 - t1) / (repeats * len(samples) * 10**6)))
  order = vals2RankedNum(rx, asIs.sd*the.Cohen)
  for k in rx:
    v = order[k]
    report[k] = f"{v.meta.rank} {v.mu:.2f} "
  report["name"]=name
  print("#"+str(list(report.keys())))
  print(list(report.values()))

#--------- --------- --------- --------- --------- --------- ------- -------
regx = r"-\w+\s*(\w+).*=\s*(\S+)"
the  = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
random.seed(the.rseed)

if __name__ == "__main__":  main()
