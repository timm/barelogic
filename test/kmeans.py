import sys
sys.path.append("..")
from src.bl import Data,csv,clone,add,some,neighbors,mids,xdist,Num,adds

def kmeans(data,k=10,r=10):
   centroids = some(data.rows, k=k)
   for _ in range(r):
      datas = {}
      errs  = {}
      for row in data.rows:
         mid      = neighbors(row, centroids, data)[0]
         err      = xdist(mid,row, data)
         x        = id(mid)
         errs[x]  = errs.get(x) or Num()
         datas[x] = datas.get(x) or clone(data)
         add(err, errs[x])
         add(row, datas[x])
      centroids = [mids(data) for data in datas.values()]
   return adds([num.mu for num in errs.values()])

print("# k    r     mu     sd")
print("#--   --   ----   ----")
for k in [2,4,8,16,32]:
   for r in [5,10,15]:
      nums = kmeans(Data(csv("data/soybean.csv")),k=k,r=r)
      print(f"{k:3}  {r:3}   {nums.mu:.2f}   {nums.sd:.2f}")
