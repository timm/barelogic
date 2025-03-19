Sumamrized from comments by 
Hayden Jones https://qr.ae/pYDdJU

## Lua Review: S.U.R.E.

Lua is a fantastic programming language that I highly recommend. Here's a review focusing on its key strengths, which I call S.U.R.E.:

- **Simplicity**: Lua code generally does what it looks like it does. It encourages building solutions piece by piece, making it easier to compose and maintain code.
- **Understandability**: Lua typically has only two levels of abstraction - tables, functions, and values at the basic level, and metatables/metamethods at a more advanced level. This makes it relatively straightforward to keep track of what's happening in your code.
- **Really fast**: Lua outperforms many interpreted languages like Python, Ruby, and Perl. LuaJIT takes this further with its JIT compiler.
- **Extensibility**: Lua is designed to be embedded and extended. It has a package manager (LuaRocks) and can be reimplemented in other languages (e.g., Gopher-lua for Go, Luerl for Erlang).

### Examples

1. **Simple Table Indexing with Metamethods**:
   ```lua
   mt = {__index = function () return math.random(10) end}
   t = {}
   setmetatable(t, mt)

   for i=1, 100 do
     print(t[i])
   end
   ```

2. **Global Table Access**:
   ```lua
   print('hello')
   -- is equivalent to
   _G.print('hello')
   ```

Lua is great for beginners and experts alike, used in various applications from game development to embedded systems.

