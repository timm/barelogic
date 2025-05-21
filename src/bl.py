# todo: 
# 1.change guards to lt, "gt". have col name in thre explictedyl
# 2. return nest from rile

"""
bl.py : barelogic, XAI for active learning + multi-objective optimization
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  

OPTIONS:  

      -a acq        xploit or xplore or adapt   = xploit  
      -b bootstraps num of bootstrap samples    = 512
      -B BootConf   bootstrap threshold         = 0.95
      -B BootConf   bootstrap threshold         = 0.95
      -c cliffConf  cliffs' delta threshold     = 0.197
      -C Cohen      Cohen threshold             = 0.35
      -d decs       decimal places for printing = 3  
      -f file       training csv file           = ../test/data/auto93.csv  
      -F Few        search a few items in a list = 50
      -g guess      size of guess               = 0.5  
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

def Num(txt=" ", at=0):
  return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG, 
           rank=0, # used by the stats functions, ignored otherwise
           goal = 0 if str(txt)[-1]=="-" else 1)

def Sym(txt=" ", at=0):
  return o(it=Sym, txt=txt, at=at, n=0, has={})

def Cols(names):
  cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
  for n,s in enumerate(names):
    col = (Num if first(s).isupper() else Sym)(s,n)
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
    i = i or (Num() if isNum(x) else Sym())
    add(x,i)
  return i

def sub(v,i,  n=1): return add(v,i,n=n,flip=-1)

def add(v,i,  n=1,flip=1): # n only used for fast sym add
  def _sym(): 
    i.has[v] = flip * n  + i.has.get(v,0)
  def _data(): 
    if not i.cols: i.cols = Cols(v)  # called on first row
    elif flip < 0:# row subtraction managed elsewhere; e.g. see eg_addSub  
       [sub(v[col.at],col,n) for col in i.cols.all]  
    else:
       i.rows += [[add( v[col.at], col,n) for col in i.cols.all]]
  def _num():
    i.lo  = min(v, i.lo)
    i.hi  = max(v, i.hi)
    if flip < 0 and i.n < 2: 
      i.mu = i.sd = 0
    else:
      d     = v - i.mu
      i.mu += flip * (d / i.n)
      i.m2 += flip * (d * (v -   i.mu))
      i.sd  = 0 if i.n <=2  else (max(0,i.m2)/(i.n-1))**.5

  if v != "?":
    i.n += flip * n 
    _sym() if i.it is Sym else (_num() if i.it is Num else _data())
  return v

#--------- --------- --------- --------- --------- --------- ------- -------
def norm(v, col):
   if v=="?" or col.it is Sym: return v
   return (v - col.lo) / (col.hi - col.lo + 1/BIG)

def mid(col): 
  return col.mu if col.it is Num else max(col.has,key=col.has.get)

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
def actLearn(data, shuffle=True):
  def _guess(row): 
    return _acquire(n/the.Stop, like(row,best,n,2), like(row,rest,n,2))
  def _acquire(p, b,r): 
    b,r = math.e**b, math.e**r
    q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
    return (b + r*q) / abs(b*q - r + 1/BIG) 

  if shuffle: random.shuffle(data.rows)
  n     =  the.start
  todo  =  data.rows[n:]
  br    = clone(data, data.rows[:n])
  done  =  ydists(data.rows[:n], br)
  cut   =  round(n**the.guess)
  best  =  clone(data, done[:cut])
  rest  =  clone(data, done[cut:])
  while len(todo) > 2  and n < the.Stop:
    n      += 1
    hi, *lo = sorted(todo[:the.Few*2], key=_guess, reverse=True)
    todo    = lo[:the.Few] + todo[the.Few*2:] + lo[the.Few:]
    add(hi, best)
    add(hi, br)
    best.rows = ydists(best.rows, br)
    if len(best.rows) >= round(n**the.guess):
      add( sub(best.rows.pop(-1), best), rest)
  return o(best=best, rest=rest, todo=todo)

#--------- --------- --------- --------- --------- --------- ------- -------
def cuts(rows, col,Y,Klass=Num):
  def _v(row) : return row[col.at]
  def _upto(x): return f"{col.txt} <= {x} ", lambda z:_v(z)=="?" or _v(z)<=x
  def _over(x): return f"{col.txt} >  {x} ", lambda z:_v(z)=="?" or _v(z)>x
  def _eq(x)  : return f"{col.txt} == {x} ", lambda z:_v(z)=="?" or _v(z)==x
  def _sym():
    n,d = 0,{}
    for row in rows:
      x = _v(row) 
      if x != "?":
        d[x] = d.get(x) or Klass()
        add(Y(row), d[x])
        n = n + 1
    return o(entropy= sum(v.n/n * spread(v) for v in d.values()),
            decisions= [_eq(k) for k,v in d.items()],
            colSplit=col)

  def _num():
    out,b4 = None,None 
    lhs, rhs = Klass(), Klass()
    xys = [(_v(r), add(Y(r),rhs)) for r in rows if _v(r) != "?"]
    xpect = spread(rhs)
    for x,y in sorted(xys, key=lambda xy: first(xy)):
      if the.leaf <= lhs.n <= len(xys) - the.leaf: 
        if x != b4:
          tmp = (lhs.n * spread(lhs) + rhs.n * spread(rhs)) / len(xys)
          if tmp < xpect:
            xpect, out = tmp,[_upto(b4), _over(b4)]
      add(sub(y, rhs),lhs)
      b4 = x
    if out:
      return o(entropy=xpect, decisions=out, colSplit=col)

  return _sym() if col.it is Sym else _num()

#--------- --------- --------- --------- --------- --------- ------- -------
def tree(rows,data,Klass=Num,xplain="",decision=lambda _:True):
   def Y(row): return ydist(row,data)
   node        = clone(data,rows)
   node.ys     = yNums(rows,data).mu
   node.impurity = yNums(rows,data).sd
   node.kids   = []
   node.decision  = decision
   node.xplain = xplain
   node.colSplit = o(txt="")
   if len(rows) >= the.leaf:
     splits=[]
     for col in data.cols.x:
       if tmp := cuts(rows,col,Y,Klass=Klass): 
        splits += [tmp]
     if splits:
       best_split = sorted(splits, key=lambda cut:cut.entropy)[0]
       for xplain,decision in best_split.decisions:
         rows1= [row for row in rows if decision(row)]
         if the.leaf <= len(rows1) < len(rows):
           node.impurity = best_split.entropy
           node.colSplit = best_split.colSplit
           node.kids += [tree(rows1,data,Klass=Klass,xplain=xplain,decision=decision)]
   return node   

def nodes(node,lvl=0, key=None):
  yield lvl,node
  for kid in (sorted(node.kids, key=key) if key else node.kids):
    for node1 in nodes(kid, lvl+1, key=key):
      yield node1

def showTree(tree, key=lambda z:z.ys):
  stats = yNums(tree.rows,tree)
  win = lambda x: 100-int(100*(x-stats.lo)/(stats.mu - stats.lo))
  print(f"{'d2h':>4} {'win':>4} {'n':>4}  {'Impurity':>5}")
  print(f"{'----':>4} {'----':>4} {'----':>4}  {'----':>5}")
  for lvl, node in nodes(tree,key=key):
    leafp = len(node.kids)==0
    post= ";" if leafp else ""
    print(f"{node.ys:4.2f} {win(node.ys):4} {len(node.rows):4} {node.impurity:5.2f}   {(lvl-1) * '|  '}{node.xplain}" + post)

def leaf(node, row):
  for kid in node.kids or []:
    if kid.decision(row): 
      return leaf(kid,row)
  return node 

def treeMDI(node, lvl=0):
  if not node.kids: return 0
  kidMDIs = 0
  for kid in node.kids:
    kidMDIs += treeMDI(kid, lvl+1)
  w = sum(kid.n for kid in node.kids)
  return kidMDIs + sum( (kid.n/w) * kid.impurity for kid in node.kids )
    
def treeFeatureImportance(tree, importance = {}):
  imp = importance or {j.txt:0 for j in tree.cols.x}
  for _, node in nodes(tree):
    if len(node.kids)>0:
      w = sum(kid.n for kid in node.kids)
      imp[node.colSplit.txt] += sum( (kid.n/w) * kid.impurity for kid in node.kids )
  return imp

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
  def _sample(vals,txt=" "): return o(vals=vals, num=adds(vals,Num(txt=txt)))
  def _same(b4,now):         return (abs(b4.num.mu - now.num.mu) < eps or
                                    cliffs(b4.vals, now.vals) and 
                                    bootstrap(b4.vals, now.vals))
  tmp,out = [],{}
  for now in sorted(_samples(), key=lambda z:z.num.mu, reverse=reverse):
    if tmp and _same(tmp[-1], now): 
      tmp[-1] = _sample(tmp[-1].vals + now.vals)
    else: 
      tmp += [ _sample(now.vals) ]
    now.num.rank = chr(96+len(tmp))
    out[now.num.txt] = now.num 
  return out

#--------- --------- --------- --------- --------- --------- ------- -------
def isNum(x): return isinstance(x,(float,int))

def first(lst): return lst[0] 

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
      if arg == "-"+first(k): 
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
                          for k,v in x.items() if first(str(k)) !="_"])+"}"
  return str(x)

def main():
  cli(the.__dict__)
  for n,s in enumerate(sys.argv):
    if fun := globals().get("eg" + s.replace("-","_")):
      arg = "" if n==len(sys.argv) - 1 else sys.argv[n+1]
      random.seed(the.rseed)
      fun(coerce(arg))

#--------- --------- --------- --------- --------- --------- ------- -------
def kpp(i, k, rows=None):
    def D(x, y):
      key = tuple(sorted((id(x), id(y))))
      if key not in mem: mem[key] = i.xdist(x,y)
      return mem[key]
     
    row, *rows = random.shuffle(rows or i.rows)[:the.some]
    out, mem = [row], {}
    for _ in range(1, k):
      dists = [min(D(x, y)**2 for y in out) for x in rows]
      r     = random.random() * sum(dists)
      for j, d in enumerate(dists):
        r -= d
        if r <= 0:
          out.append(rows.pop(j))
          break
    return out, mem
#--------- --------- --------- --------- --------- --------- ------- -------
def eg__the(_): print(the)

def eg__cols(_): 
  s="Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+"
  [print(col) for col in Cols(s.split(",")).all]

def eg__csv(file): 
  rows =list(csv(file or the.file))
  assert 3192 == sum(len(row) for row in rows)
  for row in rows[1:]: assert type(first(row)) is int

def eg__data(file):
  data=Data(csv(file or the.file))
  assert 3184 == sum(len(row) for row in data.rows)
  for row in data.rows: assert type(first(row)) is int
  [print(col) for col in data.cols.all]
  nums = adds(ydist(row,data) for row in data.rows)
  print(o(mu=nums.mu, sd=nums.sd))

def eg__ydist(file):
  data=Data(csv(file or the.file))
  r = data.rows[1] # ydist(data.rows[1],data))
  print(show(r),  ydist(r,data),the.p)
  #print(sorted(round(ydist(row,data),2) for row in data.rows))

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
   n  = 50
   b4 = [G(10,1) for _ in range(n)]
   d  = 0
   while d < 2:
     now = [x+d*R() for x in b4]
     b1  = cliffs(b4,now)
     b2  = bootstrap(b4,now)
     showd(o(d=d,cliffs=c(b1), boot=c(b2), agree=c(b1==b2)))
     d  += 0.1

def eg__rank(_):
   G  = random.gauss
   n=100
   d=dict(asIs  = [G(10,1) for _ in range(n)],
          copy1 = [G(20,1) for _ in range(n)],
          now1  = [G(20,1) for _ in range(n)],
          copy2 = [G(40,1) for _ in range(n)],
          now2  = [G(40,1) for _ in range(n)])
   for k,num in vals2RankedNums(d,the.Cohen).items():
      showd(o(what=num.txt, rank=num.rank, num=num.mu))

def eg__actLearn(file,  repeats=30):
  file = file or the.file
  name = re.search(r'([^/]+)\.csv$', file).group(1)
  data = Data(csv(file))
  b4   = yNums(data.rows,data)
  now  = Num()
  t1   = time.perf_counter_ns()
  for _ in range(repeats):
    add(ydist(first(actLearn(data, shuffle=True).best.rows ) ,data), now)
  t2  = time.perf_counter_ns()
  print(o(win= (b4.mu - now.mu) /(b4.mu - b4.lo),
          rows=len(data.rows),x=len(data.cols.x),y=len(data.cols.y),
          lo0=b4.lo, mu0=b4.mu, hi0=b4.hi, mu1=now.mu,sd1=now.sd,
          ms = int((t2-t1)/repeats/10**6),
          stop=the.Stop,name=name))

def eg__fast(file):
  def rx1(data):
    return ydist( first(actLearn(data,shuffle=True).best.rows), data)
  experiment1(file or the.file,
              repeats=30, 
              samples=[64,32,16,8],
              fun=rx1)

def eg__quick(file):
  def rx1(data):
    return [ydist(first(actLearn(data, shuffle=True).best.rows), data)]
  experiment1(file or the.file,
              repeats=10, 
              samples=[40,20,16,8],
              fun=rx1)

def eg__acts(file):
  def rx1(data):
    return [ydist(first(actLearn(data, shuffle=True).best.rows), data)]
  experiment1(file or the.file,
              repeats=30, 
              samples=[200,100,50,40,30,20,16,8],
              fun=rx1)

def experiment1(file, 
                repeats=30, samples=[32,16,8],
                fun=lambda d: ydist(first(actLearn(d,shuffle=True).best.rows),d)):
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
  order = vals2RankedNums(rx, asIs.sd*the.Cohen)
  for k in rx:
    v = order[k]
    report[k] = f"{v.rank} {v.mu:.2f} "
  report["name"]=name
  print("#"+str(list(report.keys())))
  print(list(report.values()))

def fname(f): return re.sub(".*/", "", f)

def eg__tree(file):
  data = Data(csv(file or the.file))
  model  = actLearn(data)
  b4  = yNums(data.rows,data)
  now = yNums(model.best.rows,data)
  nodes = tree(model.best.rows + model.rest.rows,data)
  print("\n"+fname(file or the.file))
  showd(o(mu1=b4.mu, mu2=now.mu,  sd1=b4.sd, sd2=now.sd))
  showTree(nodes)

def eg__rules(file):
  data  = Data(csv(file or the.file))
  b4    = yNums(data.rows, data)
  model = actLearn(data)
  now   = yNums(model.best.rows, data)
  nodes = tree(model.best.rows + model.rest.rows,data)
  todo  = yNums(model.todo, data)
  guess = sorted([(leaf(nodes,row).ys,row) for row in model.todo],key=first)
  mid = len(guess)//5
  after = yNums([row2 for row1 in model.todo for row2 in leaf(nodes,row1).rows],data)
  print(fname(file or the.file))
  print(o(txt1="b4", txt2="now",  txt3="todo",  txt4="after"))
  print(o(mu1=b4.mu, mu2=now.mu,  mu3=todo.mu,  mu4=ydist(guess[mid][1],data)))
  print(o(lo1=b4.lo, lo2=now.lo,  lo3=todo.lo,  lo4=ydist(guess[0][1],data)))
  print(o(hi1=b4.hi, hi2=now.hi,  hi3=todo.hi,  hi4=ydist(guess[-1][1],data)))
  print(o(n1=b4.n,   n2=now.n,    n3=todo.n,    n4=after.n))

def eg__afterDumb(file) : eg__after(file,repeats=30, smart=False)

def eg__after(file,repeats=30, smart=True):
  data  = Data(csv(file or the.file))
  b4    = yNums(data.rows, data) 
  overall= {j:Num() for j in [256,128,64,32,16,8]}
  for Stop in overall:
    the.Stop = Stop
    after = {j:Num() for j in [20,15,10,5,3,1]}
    learnt = Num()
    rand =Num()
    for _ in range(repeats):
      model = actLearn(data,shuffle=True)
      nodes = tree(model.best.rows + model.rest.rows,data)
      add(ydist(model.best.rows[0],data), learnt)
      guesses = sorted([(leaf(nodes,row).ys,row) for row in model.todo],key=first)
      for k in after:
        if smart:
              smart = min([(ydist(guess,data),guess) for _,guess in guesses[:k]], 
                           key=first)[1]
              add(ydist(smart,data),after[k]) 
        else:
              dumb = min([(ydist(row,data),row) for row in random.choices(model.todo,k=k)],
                   key=first)[1]
              add(ydist(dumb,data),after[k]) 
    def win(x): return str(round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo))))
    print(the.Stop, win(learnt.mu), 
          " ".join([win(after[k].mu) for k in after]), 
          1, 
          fname(file or the.file), "smart" if smart else "dumb")

#--------- --------- --------- --------- --------- --------- ------- -------

regx = r"-\w+\s*(\w+).*=\s*(\S+)"
the  = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
random.seed(the.rseed)

if __name__ == "__main__":  main()
