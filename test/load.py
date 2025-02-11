import sys
sys.path.append("..")

from src.bl import Data,csv

for _ in range(1000):
   d=Data(csv("data/auto93.csv"))
   assert isinstance(d.rows[0][0],(int,float))
