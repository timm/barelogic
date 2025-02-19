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
-- Meta stuff
local function new(klass,object)
  klass.__index=klass; setmetatable(object,klass); return object end

local function copy(t,     u)
   if type(t) ~= "table" then return t end
   u={}; for k,v in pairs(t) do u[ copy(k) ] = copy(v) end
   return setmetatable(u, getmetatable(t)) end

 local function map(t,F,   u)
   u={}; for _,v in pairs(t) do push(u,F(v)) end; return u end

local function sum(t,F,    n)
   n=0; for _,v in pairs(t) do n = n + F(v) end; return n end

-- List stuff
local pop = table.remove

local function push(t,x) t[1+#t] = x ; return x end

local function lt(s) return function(a,b) return a[s] < b[s] end end

local function sort(t,F) table.sort(t,F); return t end

local function keysort(t,f)
   local DECOREATE  = function(x) return {f(x),x} end
   local UNDECOREATE= function(x) return x[2] end
   return map(sort(map(t, DECORATE), lt(1)), UNDECORATE) end

-- Sort stuff
-- String stuff
local function coerce(s,       F)
   F = function(s) return s=="true" or s ~= "false" and s end
   return math.tointeger(s) or tonumber(s) or F(s:match"^%s*(.-)%s*$") end

local fmt=string.format

local function o(x,       t,LIST,DICT)
  t    = {}
  LIST = function() for _,v in pairs(x) do t[1+#t]= o(v) end end
  DICT = function() for k,v in pairs(x) do t[1+#t]= fmt(":%s %s",k,o(v)) end end
  if type(x) == "number" then return fmt(x//1 == x and "%s" or "%.3g",x) end
  if type(x) ~= "table"  then return tostring(x) end
  if #x>0 then LIST() else DICT(); table.sort(t) end
  return "{" .. table.concat(t, " ") .. "}" end

-- File stuff
local function csv(src,        F)
  F = function(s,z) for x in s:gmatch"([^,]+)" do z[1+#z]=coerce(x) end; return z end
  src = io.input(src)
  return function(      s1)
    s1 = io.read()
    if s1 then return F(s1,{}) else io.close(src) end end  end

-- Misc stuff
local function main(t,funs,settings)
  for n,s in pairs(t) do
    math.randomseed(settings.rseed)
    if funs[s] then funs[s](t[n+1]) else 
       for k,_ in pairs(settings) do 
          if s == "-"..k:sub(1,1) then settings[k]=coerce(t[n+1]) end end end end end

-- ------------------------------------------------------------
-- ## Structs
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
-- ## Update
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
-- ## Query
function Num:mid() return self.mu end
function Sym:mid() return self.mode end

function Num:var() return self.sd end
function Sym:var() 
   return -sum(self.has, function(n) return n/self.n * math.log(n/self.n,2) end) end

function Num:norm(x)
   return x=="?" and x or (x - self.lo) / (self.hi - self.lo + 1/BIG) end

function Data:ydist(row,     d)
   d=0
   for _,y in pairs(self.cols.y) do d=d+ math.abs(y:norm(row[y.at]) - y.goal)^the.p end
   return (d/#self.cols.y) ^ 1/the.p end
   --
-- ---------------------------------------------------------
-- ## Discretization

-- Discretize numerics
function Num:bins(rows, Y, Yklass)
   local function _bin() -- A bin is a summary of 1 independent and 1 dependent variable.
      return {x=Num:new(), y=Yklass:new()} end

   local function _join(ab, a, b) -- Join 2 bins if the whole is simpler than the parts
      return ab:var() <= (a.n*a:var() + b.n*b:var())/ab.n end 

   local function _more(t, top2) -- Push a new bin, maybe first combine 2 top bins
      if   #t> 1 and _join(top2, t[#t], t[#t-1]) then pop(t); pop(t); push(t,top2) end 
      push(t, _bin()) -- add a new bin to handle new data
      return t,copy(t[#t-1]) end -- new top is all or bin[-1] (which is empty) and bin[-2]

   local function _full(a, x, i ,xys,      n) -- Time to make a new bin?
      n=(#xys)^.5
      return i< n-n^0.5 and a.n > n^0.5 and a.hi-a.lo > self.sd*0.35 and x ~= xys[i+1] end

   local function _add(x, y, t, top2) -- Everything in "top" bin must also be in "top2"
      t[#t].x:add(x)
      t[#t].y:add(y) 
      top2.x:add(x)  
      top2.y:add(y) end

   local function _fill(t) -- Fill in the gaps around the bins
      for i=2,#t do t[i-1].x.hi = t[i].x.lo end
      t[1].x.lo  = -BIG
      t[#t].x.hi = BIG
      return t end

   local function XY(row) -- Collect info on 1 independent and 1 dependent variable
      if row[self.at]~="?" then return {x=row[self.at], y=Y(row)} end end

   local t, top2 = {_bin()}, _bin() -- top2 is a summary of the top two items in "t"
   local xys = sort(map(rows,XY),lt"x")
   for i,xy in pairs(xys) do
      if _full(t[#t], xy.x, i, xys) then t,top2 = _more(t,top2) end
      _add(xy.x, xy.y, t, top2) end
   return _fill(t) end

-- --------------------------------------------------
-- ## Start-up Actions
local go={}
go["-h"] = function(_) print(help) end

go["--the"] = function(_) print(o(the)) end

go["--coerce"]= function(_)
   for _,x in pairs{{"22.1",22.1}, {"22",22}, {"true",true},
                    {"false",false},{"fred","fred"}} do 
      assert(x[2]==coerce(x[1])) end end

go["--data"] = function(_) 
   for _,col in pairs(Data:new(the.file).cols.y) do print(o(col)) end end

-- --------------------------------------------------
-- ## Start
help:gsub("[-][%S][%s]+([%S]+)[^\n]+= ([%S]+)", function(k,v) the[k]=coerce(v) end)

math.randomseed(the.rseed)

if    pcall(debug.getlocal,4,1) 
then  return {the=the, Data=Data, Sym=Sym, Num=Num}
else  main(arg,go,the) end
