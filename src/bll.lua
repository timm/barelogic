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
-- List stuff
local pop = table.remove

local function push(t,x) t[1+#t] = x ; return x end

-- Meta stuff
local function new(klass,object)
  klass.__index=klass; setmetatable(object,klass); return object end

local function copy(t)
   if type(t) ~= "table" then return t end
   local u={}; for k,v in pairs(t) do u[ copy(k) ] = copy(v) end
   return setmetatable(u, getmetatable(t)) end

 local function map(t,F)
   F = F or function(x) return x end
   local u={}; for _,v in pairs(t) do push(u,F(v)) end; return u end

local function kap(t,F,...)
   local u={}; for k,v in pairs(t) do u[k] = F(v,k,...) end; return u end

local function sum(t,F)
   local n=0; for _,v in pairs(t) do n = n + F(v) end; return n end

-- Sort stuff
local function lt(s) return function(a,b) return a[s] < b[s] end end

local function sort(t,F) table.sort(t,F); return t end

local function keysort(t,F)
   local DECORATE  = function(x) return {F(x),x} end
   local UNDECORATE= function(x) return x[2] end
   return map(sort(map(t, DECORATE), lt(1)), UNDECORATE) end

-- String stuff
local function coerce(s)
   local function F(s1) return s1=="true" or s1 ~= "false" and s1 end
   return math.tointeger(s) or tonumber(s) or F(s:match"^%s*(.-)%s*$") end

local fmt=string.format

local function o(x)
  local t    = {}
  local LIST = function() for _,v in pairs(x) do t[1+#t]= o(v) end end
  local DICT = function() for k,v in pairs(x) do t[1+#t]= fmt(":%s %s",k,o(v)) end end
  if type(x) == "number" then return fmt(x//1 == x and "%s" or "%.3g",x) end
  if type(x) ~= "table"  then return tostring(x) end
  if #x>0 then LIST() else DICT(); table.sort(t) end
  return "{" .. table.concat(t, " ") .. "}" end

-- File stuff
local function csv(src)
  local function F(s,z) for x in s:gmatch"([^,]+)" do z[1+#z]=coerce(x) end; return z end
  src = io.input(src)
  return function()
    local s1 = io.read()
    if s1 then return F(s1,{}) else io.close(src) end end  end

-- Misc stuff
local function main(t,funs,settings)
  for n,s in pairs(t) do
    math.randomseed(settings.rseed)
    if funs[s] then funs[s](t[n+1]) else
       for k,_ in pairs(settings) do 
          if s == "-"..k:sub(1,1) then settings[k] = coerce(t[n+1]) end end end end  end
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

function Data:clone(src)
   local d= Data:new{self.cols.names}
   for _,row in pairs(src or {}) do d:add(row) end 
   return d end 

function Meta:new(names)
   local x,y,all,klass = {}, {}, {}, nil
   for at,txt in pairs(names) do
      local col = push(all, (txt:find"^[A-Z]" and Num or Sym):new(txt,at))
      if not txt:find"X$" then 
         push(txt:find"[!+-]$" and y or x, col)
         if txt:find"!$" then klass=col end end end
   return new(Meta,{x=x,y=y,all=all,klass=klass, names=names}) end 

function Some:new() return new(Some,{sorted=false, has={}) end



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

function Num:add(n)
  if n=="?" then return n end
  self.n  = self.n + 1
  n       = n + 0 -- ensure we have numbers
  local d = n - self.mu
  self.mu = self.mu + d/self.n
  self.m2 = self.m2 + d*(n - self.mu)
  self.sd = self.n < 2 and 0 or (self.m2/(self.n - 1))^0.5
  self.lo = math.min(n, self.lo)
  self.hi = math.max(n, self.hi) end

function Some:add(n) self.sorted=false; push(self.has,n) end

function Some:ok()
   if not self.sorted then table.sort(self.has) end
   self.sorted=true
   return self end

function Some:mid() t=self:ok().has; return t[#t//2] end

function Some:var() 
   t = self:ok().has 
   c = self:mid()
   if #t > 20 then a,b = t[math.max(1,#t*0.05 //1)], t[#t*0.95//1] 
              else a,b = t[1], t[#t] end
   return ((a^2 + b^2 + c^2 - a*b - a*c - b*c)/18)^0.5 end

-- ---------------------------------------------------------
-- ## Misc Query
function Num:mid() return self.mu end
function Sym:mid() return self.mode end

function Num:var() return self.sd end
function Sym:var() 
   local function F(n) return n/self.n * math.log(n/self.n,2) end
   return -sum(self.has, F) end 

function Num:norm(x)
   return x=="?" and x or (x - self.lo) / (self.hi - self.lo + 1/BIG) end

function Data:ydist(row)
   local function F(col) return math.abs(col:norm(row[col.at]) - col.goal)^the.p end
   return (sum(self.cols.y, F) / #self.cols.y) ^ (1/the.p) end

function Data:ysort()
   local function F(row) return self:ydist(row) end 
   self.rows = keysort(self.rows, F)
   return self end 

function Some.merge(i,j,   k)
   k=copy(i); for _,n in pairs(j.has) do k:add(n) end; k.sorted=false; return k end

-- ---------------------------------------------------------
-- ## Discretization
function Bin:new(txt,at,lo,hi) 
   return new(Bin,{txt=txt,at=at,lo=lo,hi=hi or lo,ys=Some:new()}) end

function Bin:add(x,y)
   self.ys:add(y)
   self.lo = math.min(x, self.lo)
   self.hi = math.max(x, self.hi)  end

function Bin,merged(i,j,n)
  local k,n1,n2,n3,v1,v2,v3
  k         = copy(i)
  k.has     = i.has:merge(j.has)
  n1,n2,n12 = #i.ys.has, #j.ys.has, #k.ys.has
  v1.v2.v12 = i.has:var(), j.has:var(), k.has:var()
  if n1 < n or n2 < n or v12 <= (v1*n1 + v2*n2) / n12 then return k end end

function Sym:discretize(x) return x end

function Num:discretize(x) return self:norm(x) * the.bins // 1

function Data:bins(rows)
   local function _bins(col)
      local n,tmp,n,j = 0,{}
      for i,row in pairs(rows) do
          x =  row[self.at]
          if x ~= "?" then
             n = n + 1
             k = col:discretize(x)
             tmp[k] = tmp[k] or Bin(col.txt.col.at,lo)
             tmp[k]:add(x, self:ydist(row)) end  end
      return col:merge(sort(map(tmp),lt"lo"),n^0.5) end 

   return map(self.cols.x, _bins) end 

function Sym:merge(bins,_) return bins end

function Num:merge(bins0,n)
   local i,bins = 0,{}
   while i <= #bins0 do
      i = i + 1 
      local bin = bins0[i]
      if i < #bins0-1 then
         local merged = bin:merged(bins0[i+1],n)
         if merged then bin,i = merged, i+1 end end
      push(bins,bin)
   end
   if #bins < #bins0 then return self:merge(bins,n) else
      bins[1].lo = -BIG
      bins[#bins].hi = BIG
      return bins end end

-- ---------------------------------------------------------
function Data:cuts(rows)
   local function F(bin) 
      n=Num:new()
      for _,y in pairs(bin.ys) do n:add(y) end
      return n:var()*n.n/#rows end

   return keysort(self:bins(rows),function(bins4col) return sum(bins4col,F) end)[1] end

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
   for _,col in pairs(Data:new(the.file).cols.y) do 
      print(o(col)) end end

go["--bins"] = function(_)
   local d = Data:new(the.file)
   local Y = function(r) return d:ydist(r) end
   for _,p in pairs(d.cols.x[1]:bins(d.rows, Num,Y)) do
       print(o(p.x)) end end
   
-- --------------------------------------------------
-- ## Start
help:gsub("[-][%S][%s]+([%S]+)[^\n]+= ([%S]+)", function(k,v) the[k]=coerce(v) end)

math.randomseed(the.rseed)

if    pcall(debug.getlocal,4,1) 
then  return {the=the, Data=Data, Sym=Sym, Num=Num}
else  main(arg,go,the) end
