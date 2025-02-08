import sys
sys.path.append("..")

from src.bl import the, Data,csv,mid,spread,showd

d=Data(csv(the.file))
print("klass ", d.cols.klass or "none")
showd({col.txt : mid(col) for col in d.cols.all})
showd({col.txt : spread(col) for col in d.cols.all})
