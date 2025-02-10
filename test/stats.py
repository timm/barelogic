import sys
sys.path.append("..")
from src.bl import the,adds,mid,delta

# Non-parametric effect size. threshold is border between small=.11 and medium=.28 
# from Table1 of  https://doi.org/10.3102/10769986025002101
def cliffs(xs: List[float], ys: List[float]) -> bool:
   n,lt,gt = 0,0,0
   for x in xs:
     for y in ys:
        n += 1
        if x > y: gt += 1
        if x < y: lt += 1 
   return abs(lt - gt)/n  < the.Cliffs # 0.197) 

# non-parametric significance test From Introduction to Bootstrap, 
# Efron and Tibshirani, 1993, chapter 20. https://doi.org/10.1201/9780429246593
def bootstrap(ys: List[float], zs: List[float] )  -> bool:
    y0,z0  = ys, zs
    x,y,z  = adds(y0+z0), adds(y0), adds(z0)
    yhat   = [y1 - mid(y) + mid(x) for y1 in y0]
    zhat   = [z1 - mid(z) + mid(x) for z1 in z0] 
    _some  = lambda lst: adds(random.choices(lst, k=len(lst))) 
    s      = the.stats.bootstraps
    n      = sum(delta(_some(yhat), _some(zhat))  > delta(y,z) for _ in range(the.boots)) 
    return n / s >= the.confidence


