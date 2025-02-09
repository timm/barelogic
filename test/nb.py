import sys
sys.path.append("..")
from src.bl import Data,csv,clone,add,likes,the,shuffle

def nb(data):
   d, acc = {}, 0
   for j,row in enumerate(data.rows):
      x        = row[ data.cols.klass.at ]
      d[x]     = d.get(x) or clone( data )
      d[x].txt = x
      acc += x == likes(row, d.values()).txt
      add(row, d[x])
   return acc/len(data.rows)

def act(file):
  print("\nk m p\tfile\n- - --\t----")
  d = Data(csv("data/" + file + ".csv"))
  for k in range(4):
     for m in range(4):
        the.k = k
        the.m = m
        print(k,m,int(100*nb(d)),"\t" + file)

[act(f) for f in ["diabetes","soybean"]]
