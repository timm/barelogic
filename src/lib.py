import re,ast,math,random

any=random.choice
R=rabdom.random

def coerce(s):
  try: return ast.literal_eval(s)
  except Exception: return s

def show(x):
  it = type(x)
  if it == float: return str(round(x,the.decs))
  if it == list:  return ', '.join([show(v) for v in x])
  if it == dict:  return "("+' '.join([f":{k} {show(v)}" for k,v in x.items()])+")"
  if it == o:     return x.__class__.__name__ + show(x.__dict__)
  if it == str:   return '"'+str(x)+'"'
  if callable(x): return x.__name__
  return str(x)

class o:
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = show



