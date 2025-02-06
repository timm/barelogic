from lib import *

def like(lst, data, nall=100, nh=2):
  def _like1(v,col):
    if col.it is Sym:
      return (sym.has.get(v,0) + the.m*prior) / (sym.n + the.m)
    else:
      sd    = col.sd + 1/Big
      nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
      denom = (2*math.pi*sd*sd) ** 0.5
      return max(0, min(1, nom/denom))

  prior= (data.n + the.k) / (nall + the.k*nh)
  likes= [like1(lst[col.at], col) for col in data.cols.x if lst[col.at]!="?"]
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


