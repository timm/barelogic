import sys
sys.path.append("..")
from src.bl import Data,csv,ydist

d = Data(csv("data/auto93.csv"))
for j,row in enumerate(sorted(d.rows,key=lambda row: ydist(row,d))):
   if j<10 or j % 50 == 0: 
       print(j,row,round(ydist(row,d),2))
