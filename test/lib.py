import sys
sys.path.append("..")
from src.bl import ent,csv,adds,shuffle,spread

[print(j,r) for j,r in enumerate(csv("data/auto93.csv")) if not j % 30] 

[print(shuffle([1,2,3,4,5])) for _ in range(10)]

assert abs(1.3787 - spread(adds(["a","a","a","a","b","b","c"]))) < 0.01

