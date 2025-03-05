local the,help = {},[[
bll.lua : Naive Bayes    
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  
   
OPTIONS:  

      -a acq     xploit or xplore or adapt   = xplout  
      -b bins    number of bins              = 17
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
local l,Num,Sym,Data,Meta,XY = {},{},{},{},{},{}

-- ------------------------------------------------------------
-- ## Library
-- List stuff
l.pop = table.remove

function l.push(t,x) t[1+#t] = x ; return x end

-- Math stuff
l.R = math.random

function l.gaussian(  mu,sd)
   local sq,pi,log,cos = math.sqrt,math.pi,math.log,math.cos
   return (mu or 0) + (sd or 1)* sq(-2*log(l.R())) * cos(2*pi*l.R())  end

function l.weibull(x, k, lambda)
   if x < 0 or k <= 0 or lambda <= 0 then return 0 end
   local term = (x / lambda)^(k - 1)
   local exp_term = math.exp(- (x / lambda)^k)
   return (k / lambda) * term * exp_term end

-- Meta stuff
function l.new(klass,object)
   klass.__index=klass; setmetatable(object,klass); return object end

function l.copy(t)
   if type(t) ~= "table" then return t end
   local u={}; for k,v in pairs(t) do u[ l.copy(k) ] = l.copy(v) end
   return setmetatable(u, getmetatable(t)) end

function l.map(t,F,  self,      G)
   if self then G=F; F = function(v) return getmetatable(self)[G](self,v) end end
   F = F or function(v) return v end
   local u={}; for _,v in pairs(t) do l.push(u,F(v)) end; return u end

function l.sum(t,F)
   local n=0; for _,v in pairs(t) do n = n + F(v) end; return n end

-- Sort stuff
function l.lt(s) return function(a,b) return a[s] < b[s] end end

function l.sort(t,F) table.sort(t,F); return t end

function l.keysort(t,F,  self,       G)
   if self then G=F; F = function(v) return getmetatable(self)[G](self,v) end end
   local DECORATE  = function(x) return {F(x),x} end
   local UNDECORATE= function(x) return x[2] end
   return l.map(l.sort(l.map(t, DECORATE), l.lt(1)), UNDECORATE) end

-- String stuff
function l.coerce(s)
   local function F(s1) return s1=="true" or s1 ~= "false" and s1 end
   return math.tointeger(s) or tonumber(s) or F(s:match"^%s*(.-)%s*$") end

l.fmt=string.format

function l.o(x)
   local t    = {}
   local LIST = function() for _,v in pairs(x) do t[1+#t]=l.o(v) end end
   local DICT = function() for k,v in pairs(x) do t[1+#t]=l.fmt(":%s %s",k,l.o(v)) end end
   if type(x) == "number" then return l.fmt(x//1 == x and "%s" or "%.3g",x) end
   if type(x) ~= "table"  then return tostring(x) end
   if #x>0 then LIST() else DICT(); table.sort(t) end
   return "{" .. table.concat(t, " ") .. "}" end

-- Misc stuff
function l.csv(src)
   local function F(s) 
      local z={}; for x in s:gmatch"([^,]+)" do z[1+#z]=l.coerce(x) end;return z end
   src = io.input(src)
   return function()
      local s1 = io.read()
      if s1 then return F(s1) else io.close(src) end end  end

function l.main(t,funs,settings)
   for n,s in pairs(t) do
      math.randomseed(settings.rseed)
      if funs[s] then funs[s](t[n+1]) else
         for k,_ in pairs(settings) do 
            if s == "-"..k:sub(1,1) then settings[k]=l.coerce(t[n+1]) end end end end end

-- ------------------------------------------------------------
-- Namespace stuff
local R,coerce,copy,csv,fmt,gaussian,keysort,lt =
      l.R,l.coerce,l.copy,l.csv,l.fmt,l.gaussian,l.keysort,l.lt
local main,map,new,o,pop,push,sort,sum,weibull =
      l.main,l.map,l.new,l.o,l.pop,l.push,l.sort,l.sum,l.weibull

-- ------------------------------------------------------------
-- ## Structs

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

-- --------------------------------------------------------------------
-- ## Update
function Data:add(row)
   if   self.cols
   then push(self.rows, self.cols:add(row))
   else self.cols = Meta:new(row) end end

function Meta:add(row)
   for _,col in pairs(self.all) do col:add(row[col.at]) end
   return row end

function Sym:add(v,  n)
   if v ~= "?" then 
      self.n = self.n + (n or 1)
      self.has[v] = (n or 1) + (self.has[v] or 0)
      if self.has[v] > self.most then
         self.most, self.mode = self.has[v], v end  end
   return v end

function Num:add(v)
   if v ~= "?" then 
      self.n  = self.n + 1
      local d = v - self.mu
      self.mu = self.mu + d/self.n
      self.m2 = self.m2 + d*(v - self.mu)
      self.sd = self.n < 2 and 0 or (self.m2/(self.n - 1))^0.5
      self.lo = math.min(v, self.lo)
      self.hi = math.max(v, self.hi) end
   return v end

function Sym.merge(i,j,   k)
   k = copy(i)
   for v,n in pairs(j.has) do k:add(v,n) end
   return k end

function Num.merge(i,j,   k)
   k    = Num:new(i.txt, i.at)
   k.n  = i.n + j.n
   k.mu = (i.n*i.mu + j.n*j.mu)/(i.n + j.n)
   k.m2 = i.m2 + j.m2 + (i.n * j.n/(i.n+j.n)) * (i.mu - j.mu)^2
   k.sd = k.n < 2 and 0 or (k.m2/(k.n - 1))^0.5
   k.lo = math.min(i.lo,j.lo)
   k.hi = math.max(i.hi,j.hi)
   return k end 

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
   return keysort(self.rows,"ydist",self) end

-- ---------------------------------------------------------
-- ## XY
function XY:new(col,lo,hi) 
   self= new(XY,{x=Num:new(col.txt,col.at), y=nil})
   self.x.lo = lo
   self.x.hi = hi or lo
   self.y    = (type(lo)=="number" and Num or Sym):new()
   return self end

function XY:add(x,y)
   self.x:add(x)
   self.y:add(y) end

function XY.merge(i,j)
  return new(XY,{x = i.x:merge(j.x), y = i.y:merge(j.y)}) end

function XY.merges(i,j,n,eps,   k)
  k = i:merge(j)
  local n1,n2,n12 = i.x.n, j.x.n, k.x.n
  local v1,v2,v12 = i.y:var(), j.y:var(), k.y:var()
  if math.abs(i.x.mu - j.x.mu) < eps or n1<n or n2<n or v12<=(v1*n1+v2*n2)/n12 
  then return k end end

function XY:selects(rows)
   local yes,no = {},{}
   for _,row in pairs(rows) do push(self:select(row) and yes or no, row) end
   return yes,no end

function XY:select(row)
   local v = row[self.x.at]
   if v == "?" then return true end
   if self.x.lo == self.x.hi then return v == self.x.lo end
   return self.x.lo <= v and v < self.x.hi end

-- ---------------------------------------------------------
-- ## Discretization
function Data:xys(rows,Y)
   return map(self.cols.x, function(col) return self:xys1(col,rows,Y) end) end

function Data:xys1(col,rows,Y)
   local n,tmp = 0,{}
   for _,row in pairs(rows) do
      local x =  row[col.at]
      if x ~= "?" then
         local k = col:discretize(x)
         tmp[k] = tmp[k] or XY:new(col,x)
         tmp[k]:add(x, Y(row)) end end
   return col:merges(keysort(map(tmp),function(v) return v.x.lo end),
                     (#rows)^.5) end

function Sym:discretize(x) return x end
function Num:discretize(x) return self:norm(x) * the.bins // 1 end

function Sym:merges(xys,_) return xys end
function Num:merges(xys,n) 
   local function _fill(xys1)
      for i = 2,#xys1 do xys1[i-1].x.hi = xys1[i].x.lo end
      xys1[1   ].x.lo = -BIG
      xys1[#xys].x.hi =  BIG
      return xys1 end 

   local new,i = {},0
   while i <= #xys do
      i = i+1
      local a = xys[i]
      if i < #xys 
      then local merged = a:merges(xys[i+1], n, self:var()*.35)
           if merged then a,i = merged,i+1 end end
      push(new, a) end
   return #new < #xys and self:merges(new,n) or _fill(xys) end

-- ---------------------------------------------------------
function Data:cuts(rows,      X,F,D)
   X = function(xy)  return xy.y.sd * xy.y.n / #rows end
   F = function(xys) return sum(xys, X) end
   D = function(row) return self:ydist(row) end
   return keysort(self:xys(rows, D), F)[1] end

function Data:grow(  guard,stop)
  local stop = stop or (#self.rows)^.5
  self.guard = guard
  self.kids  = {}
  for _,xy in pairs(self:cuts(self.rows)) do
     local rows = self.rows
     local rows,more = xy:selects(rows)
     if #rows < #self.rows and #rows > stop then 
        push(self.kids, self:clone(rows):grow(xy, stop)) end
     rows = more end 
  return self end 

-- --------------------------------------------------
-- ## Start-up Actions
local go={}
go["--all"] = function(_)
   local n,all=0,{}
   for k,fun in pairs(go) do if k ~= "--all" then push(all,{key=k,fun=fun}) end end
   for _,p in pairs(sort(all, lt"key")) do
      local ok,err= xpcall(p.fun, debug.traceback) 
      if not ok then n=n+1; print("\27[31mERROR:\27[0m",p.key,err) end end 
   print(string.format("\n%.2f%% errors.", 100*n/#all)) end

go["-h"] = function(_) print(help) end

go["--the"] = function(_) print(o(the)) end

go["--coerce"]= function(_)
   for _,x in pairs{{"22.1",22.1}, {"22",22}, {"true",true},
                    {"false",false},{"fred","fred"}} do 
      assert(x[2]==coerce(x[1])) end end

go["--data"] = function(_) 
   for _,col in pairs(Data:new(the.file).cols.y) do 
      print(o(col)) end end

go["--ysorts"] = function(_)
   local d = Data:new(the.file)
   for i,row in pairs(d:ysort()) do 
      if i % 30 == 1 then print(fmt("%3s,  %.3f,  %s",i,d:ydist(row),o(row))) end end end
   
go["--nums"] = function(_)
   for _ = 1,10^3 do -- 100 times, compare two ways to calc sd
      local n1,n2,n12 = Num:new(), Num:new(), Num:new()
      local r = function() return R(0,10^6)/(10^6) end
      local k1, l1 = 0.5+r(), 5*r() 
      local k2, l2 = 0.5+r(), 5*r() 
      for _ = 1,10^2 do
         n12:add(n1:add(weibull(2.5*r(), k1,l1)))
         n12:add(n2:add(weibull(2.5*r(), k2,l2))) end
      assert(10^-6 > math.abs(1 - n12.sd/ (n1:merge(n2).sd))) end end
  
go["--xys"] = function(_)
   local d = Data:new(the.file)
   local Y = function(row) return d:ydist(row) end
   for n,xys in pairs(d:xys(d.rows, Y)) do 
      print""
      for _,xy in pairs(xys) do print(n,o(xy)) end end 
   print""
   for _,xy in pairs(d:cuts(d.rows)) do print(0,o(xy)) end end

-- --------------------------------------------------
-- ## Start

help:gsub("[-][%S][%s]+([%S]+)[^\n]+= ([%S]+)", function(k,v) the[k]=coerce(v) end)
math.randomseed(the.rseed)

if not pcall(debug.getlocal,4,1) then  main(arg,go,the) end
return {the=the, Data=Data, Sym=Sym, Num=Num}
