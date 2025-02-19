local the,help = {},[[
nb.lua : Naive Bayes    
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  
   
OPTIONS:  

      -a acq     xploit or xplore or adapt   = xplout  
      -d decs    decimal places for printing = 2  
      -f file    training csv file           = ../test/data/auto93.csv  
      -g guess   size of guess               = 0.5  
      -G Guesses max number of guesses       = 100  
      -k k       low frequency Bayes hack    = 1  
      -l leaf    min size of tree leaves     = 2
      -m m       low frequency Bayes hack    = 2  
      -p p       distance formula exponent   = 2  
      -r rseed   random number seed          = 1234567891  
      -s start   where to begin              = 4  
      -S Stop    where to end                = 32]]

local BIG=1E32

-- ------------------------------------------------------------
-- ## Library
-- ### Meta
local function coerce(s,       F)
   F = function(s) return s=="true" or s ~= "false" and s end
   return math.tointeger(s) or tonumber(s) or F(s:match"^%s*(.-)%s*$") end

-- ### Polymorphism
local function new(klass,object)
  klass.__index=klass; setmetatable(object,klass); return object end

-- ### Lists
local function push(t,x) t[1+#t] = x ; return x end

local function lt(s) return function(a,b) return a[s] < b[s] end end

local function copy(t,     u)
   if type(t) ~= "table" then return t end
   u={}; for k,v in pairs(t) do u[ copy(k) ] = copy(v) end
   return setmetatable(u, getmetatable(t)) end

-- ### File
local function csv(src,        F)
  F = function(s,z) for x in s:gmatch"([^,]+)" do z[1+#z]=coerce(x) end; return z end
  src = io.input(src)
  return function(      s1)
    s1 = io.read()
    if s1 then return F(s1,{}) else io.close(src) end end  end

-- ### String
local fmt=string.format

local function o(x,       t,LIST,DICT)
  t    = {}
  LIST = function() for _,v in pairs(x) do t[1+#t]= o(v) end end
  DICT = function() for k,v in pairs(x) do t[1+#t]= fmt(":%s %s",k,o(v)) end end
  if type(x) == "number" then return fmt(x//1 == x and "%s" or "%.3g",x) end
  if type(x) ~= "table"  then return tostring(x) end
  if #x>0 then LIST() else DICT(); table.sort(t) end
  return "{" .. table.concat(t, " ") .. "}" end

-- ### Misc
local function main(t,funs,settings)
  for n,s in pairs(t) do
    math.randomseed(settings.rseed)
    if funs[s] then funs[s](t[n+1]) else 
       for k,_ in pairs(settings) do 
          if s == "-"..k:sub(1,1) then settings[k]=coerce(t[n+1]) end end end end end

-- ------------------------------------------------------------
local Num,Sym,Data,Meta={},{},{},{}

function Num:new(txt,at)
   return new(Num,{txt=txt or " ", at=at or 0, n=0, 
                   mu=0, sd=0, m2=0, hi= -BIG, lo= BIG,
                   goal = tostring(txt):find"-$" and 0 or 1}) end  

function Sym:new(txt, at)
   return new(Sym, {txt=txt or "", at=at or 0, n=0, has={}, most=0, mode=nil}) end

function Data:new(src)
   self = new(Data,{rows={}, cols=nil})
   if   type(src)=="string" 
   then for   row in csv(src)         do self:add(row) end
   else for _,row in pairs(src or {}) do self:add(row) end end 
   return self end

function Data:clone(src,       d)
   d= Data:new{self.cols.names}
   for _,row in pairs(src or {}) do d:add(row) end 
   return d end 

function Meta:new(names,        x,y,all,col,klass)
   x,y,all,klass = {}, {}, {}, nil
   for at,txt in pairs(names) do
      col = push(all, (txt:find"^[A-Z]" and Num or Sym):new(txt,at))
      if not txt:find"X$" then 
         push(txt:find"[!+-]$" and y or x, col)
         if txt:find"!$" then klass=col end end end
   return new(Meta,{x=x,y=y,all=all,klass=klass, names=names}) end 
-- --------------------------------------------------------------------
function Data:add(row)
   if   self.cols
   then push(self.rows, self.cols:add(row))
   else self.cols = Meta:new(row) end end

function Meta:add(row)
   for _,col in pairs(self.all) do col:add(row[col.at]) end
   return row end

function Sym:add(x)
  if x=="?" then return x end
  self.n = self.n + 1
  self.has[x] = 1 + (self.has[x] or 0)
  if self.has[x] > self.most then
    self.most, self.mode = self.has[x], x end end

function Num:add(n,       d)
  if n=="?" then return n end
  self.n  = self.n + 1
  n       = n + 0 -- ensure we have numbers
  d = n - self.mu
  self.mu = self.mu + d/self.n
  self.m2 = self.m2 + d*(n - self.mu)
  self.sd = self.n < 2 and 0 or (self.m2/(self.n - 1))^0.5
  self.lo = math.min(n, self.lo)
  self.hi = math.max(n, self.hi) end

-- ---------------------------------------------------------
function Num:norm(x)
   return x=="?" and x or (x - self.lo) / (self.hi - self.lo + 1/BIG)

function Data:ydist(row,     d)
   d=0
   for _,y in self.cols.y do d=d+ math.abs(y:norm(row[y.at]) - y.goal)^the.p end
   return (d/#self.cols.y) ^ 1/the.p end
-- ---------------------------------------------------------
function NUM:bin(rows,         Thing,xys,t):
   xys = {}
   for _,row in pairs(rows) do 
     if row[self.at] ~= "?" then 
        push(xys, {x=row[self.at], y=data:ydist(row)}) end end end
   table.sort(xys,lt"x")
   Thing = Thing or (type(xys[1].y) == "number" and Num or Sym)
   local t,x,
   small = (#xy)^.5
   t={}
   push(t, {x=Num:new(),y=Thing:new()})
   for i,xy in parts(xys) do
     x,y = t[#t].x, t[#t].y
     x:add(xy.x); y:add(xy.y)
     if i < #xys - small and  x.n > small and x.hi - x.lo > self.sd*0.35 
        and xy.x ~= xys[i+1].x then
        if #t > 1 then
           t[#t-1] = {x=x:merge(t[#t=1].x), y=y:merge(t[#t-1].y)}
           t[#t]   = {x=Num:new(),y=Thing:new()}
        else
           push(t, {x=Num:new(),y=Thing:new()} end end

-- ## Actions
local go={}
go["-h"] = function(_) print(help) end

go["--the"] = function(_) print(o(the)) end

go["--coerce"]= function(_)
   for _,x in pairs{{"22.1",22.1}, {"22",22}, {"true",true},
                    {"false",false},{"fred","fred"}} do 
      assert(x[2]==coerce(x[1])) end end

go["--data"] = function(_) 
   for _,col in pairs(Data:new(the.file).cols.y) do print(o(col)) end end

-- ## Start
help:gsub("[-][%S][%s]+([%S]+)[^\n]+= ([%S]+)", function(k,v) the[k]=coerce(v) end)

math.randomseed(the.rseed)

if    pcall(debug.getlocal,4,1) 
then  return {the=the, Data=Data, Sym=Sym, Num=Num}
else  main(arg,go,the) end
