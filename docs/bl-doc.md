# Tutorial: Understanding bl.py - Barelogic for Active Learning & Multi-Objective Optimization

Welcome! This document explores `bl.py`, a Python script designed by Tim Menzies for performing Explainable AI (XAI) using active learning and multi-objective optimization on tabular data. It takes a "bare logic" approach, meaning it implements core concepts from scratch rather than relying heavily on large external libraries.

We'll break down the code step-by-step, covering the underlying math, Python concepts, data structures, and the AI algorithms used.

**Table of Contents:**

1.  [Motivation: Finding the "Best" Cars](#1-motivation-finding-the-best-cars)
2.  [Mathematical Foundations](#2-mathematical-foundations)
    * [Descriptive Statistics (Mean, Standard Deviation)](#descriptive-statistics-mean-standard-deviation)
    * [Entropy](#entropy)
    * [Normalization](#normalization)
    * [Distance Metric (p-norm)](#distance-metric-p-norm)
    * [Naive Bayes Likelihood](#naive-bayes-likelihood)
3.  [Python Notes (Beyond the Basics)](#3-python-notes-beyond-the-basics)
    * [The `o` Class and `lambda`](#the-o-class-and-lambda)
    * [List/Dictionary Comprehensions](#listdictionary-comprehensions)
    * [Generators (`yield`)](#generators-yield)
    * [`__name__ == "__main__"`](#__name____main__)
    * [Command-Line Arguments (`sys.argv`, `re`)](#command-line-arguments-sysargv-re)
    * [File Handling (`with open`)](#file-handling-with-open)
4.  [Core Data Structures](#4-core-data-structures)
    * [Columns: `Num` and `Sym`](#columns-num-and-sym)
    * [Column Roles: `x` vs `y`](#column-roles-x-vs-y)
    * [Managing Columns: `Cols`](#managing-columns-cols)
    * [The Dataset: `Data`](#the-dataset-data)
5.  [Efficient Updates: Incremental `add` and `sub`](#5-efficient-updates-incremental-add-and-sub)
    * [Why Incremental?](#why-incremental)
    * [How `add` Works](#how-add-works)
    * [How `sub` Works](#how-sub-works)
6.  [AI Tools (Simple to Complex)](#6-ai-tools-simple-to-complex)
    * [Evaluating Rows: `ydist`](#evaluating-rows-ydist)
    * [Calculating Likelihood: `like` and `likes`](#calculating-likelihood-like-and-likes)
    * [Intelligent Data Selection: `actLearn`](#intelligent-data-selection-actlearn)
7.  [Appendix: Configuration Options (`the`)](#7-appendix-configuration-options-the)

*(Note: This tutorial ignores the `XY`, `merge`, and `isMerged` functions related to discretization, as they appear incomplete in the provided code.)*

---

## 1. Motivation: Finding the "Best" Cars

Imagine you have a dataset of cars with various features:

| Cylinders | Displacement | Horsepower | Weight- | Acceleration+ | MPG+ | Model    | Origin |
| :-------- | :----------- | :--------- | :------ | :------------ | :--- | :------- | :----- |
| 8         | 307          | 130        | 3504    | 12.0          | 18   | chevy    | USA    |
| 8         | 350          | 165        | 3693    | 11.5          | 15   | buick    | USA    |
| 4         | 97           | 46         | 1835    | 20.5          | 28   | datsun   | Japan  |
| 6         | 199          | 90         | 2648    | 15.0          | 22   | amc      | USA    |
| 4         | 140          | 86         | 2790    | 15.6          | 27   | ford     | USA    |
| 4         | 98           | 80         | 2164    | 16.0          | 32   | toyota   | Japan  |
| ...       | ...          | ...        | ...     | ...           | ...  | ...      | ...    |

You want to find a *small* group of cars that are generally "the best" according to multiple criteria: low weight (`Weight-`), high acceleration (`Acceleration+`), and high miles-per-gallon (`MPG+`).

Evaluating *every single car* might be time-consuming or expensive if evaluation involves real-world tests. `bl.py` uses **active learning** to intelligently select a small number of cars to "evaluate" (in this case, by analyzing their data). It aims to quickly converge on a set of cars that represent the best trade-offs according to your defined goals (`-`, `+`).

The script might process this data and output a much smaller list, perhaps just 5-10 cars, identified as being Pareto-optimal or near-optimal based on the criteria, without needing to analyze the entire dataset exhaustively. This is achieved using the techniques described below.

---

## 2. Mathematical Foundations

`bl.py` relies on several standard mathematical and statistical concepts:

### Descriptive Statistics (Mean, Standard Deviation)

* **Mean (μ):** The average value. Calculated as `sum(values) / count(values)`.
* **Standard Deviation (σ):** Measures the spread or dispersion of data around the mean.
* **Online Calculation:** `bl.py` uses Welford's online algorithm (via the `m2` variable in `Num`) to update the mean and standard deviation incrementally as new data points arrive (`add`) or are removed (`sub`). This avoids storing all values.

    ```python
    # Simplified concept for adding 'v' to a Num 'i'
    d = v - i.mu
    i.mu += d / i.n
    i.m2 += d * (v - i.mu) # Update sum of squared differences
    i.sd = 0 if i.n <= 2 else (max(0,i.m2)/(i.n-1))**0.5
    ```

### Entropy

* For symbolic columns (`Sym`), standard deviation doesn't apply. Instead, **Entropy** is used to measure the "mixed-up-ness" or uncertainty. A column where all values are the same has zero entropy. A column with values evenly distributed has maximum entropy.
* Formula: `H = -Σ [ (p_i) * log2(p_i) ]` where `p_i` is the proportion of the i-th category.

    ```python
    def ent(d): # 'd' is the 'has' dictionary {value: count}
      N = sum(n for n in d.values())
      return -sum(n/N * math.log(n/N,2) for n in d.values())

    def spread(col): return col.sd if col.it is Num else ent(col.has)
    ```

### Normalization

* To compare values from different columns with different ranges (e.g., Horsepower 40-200 vs. Acceleration 8-25), values are normalized to a common scale, typically 0 to 1.
* Formula (Min-Max): `norm_v = (v - min) / (max - min)` (adding a small constant `1/BIG` to the denominator prevents division by zero if min equals max).

    ```python
    def norm(v, col):
      if v=="?" or col.it is Sym: return v # Ignore missing or symbolic
      # Add 1/BIG for numerical stability if hi == lo
      return (v - col.lo) / (col.hi - col.lo + 1/BIG)
    ```

### Distance Metric (p-norm)

* To evaluate how well a row meets multiple objectives (`y` columns), `bl.py` calculates a distance between the row's normalized objective values and the "ideal" values (0 for minimize `-`, 1 for maximize `+`).
* It uses the **Minkowski distance** (p-norm), generalized from Euclidean (p=2) and Manhattan (p=1) distance.
* Formula: `Distance = ( Σ | norm(value_i) - goal_i |^p )^(1/p)`
* The `the.p` parameter controls which norm is used (default is 2, Euclidean).

    ```python
    # Inside ydist(row, data):
    # 'c.goal' is 0 for '-' columns, 1 for '+' or '!' columns
    # 'the.p' is the exponent from config (default 2)
    total_dist_power_p = sum(
        abs(norm(row[c.at], c) - c.goal)**the.p
        for c in data.cols.y # Only consider objective columns
    )
    average_dist_power_p = total_dist_power_p / len(data.cols.y)
    final_dist = average_dist_power_p ** (1/the.p)
    ```

### Naive Bayes Likelihood

* Used in active learning (`like` function) to estimate the probability that a given `row` belongs to a specific cluster or `Data` subset (e.g., the `best` or `rest` set).
* **Naive Assumption:** Assumes all features (columns) are independent, which simplifies calculation but is often effective in practice.
* **Calculation:**
    1.  **Prior Probability:** The base probability of belonging to the subset, based on its size relative to the total data size (`prior = (data.n + the.k) / (nall + the.k*nh)`). Incorporates Laplace smoothing (`the.k`).
    2.  **Likelihood per Feature:**
        * **Numerical (`Num`):** Calculates the probability density of the row's value (`v`) given the subset's mean (`mu`) and standard deviation (`sd`), using the Gaussian (Normal) probability density function (PDF).
            ```python
            # Simplified from _col inside like() for Num
            sd = col.sd + 1/BIG # Avoid division by zero
            nom = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
            denom = (2*math.pi*sd*sd) ** 0.5
            likelihood_num = max(0, min(1, nom/denom)) # Clamp between 0 and 1
            ```
        * **Symbolic (`Sym`):** Calculates the probability based on the frequency of the row's value (`v`) within the subset's counts (`col.has`). Includes Laplace smoothing (`the.m`).
            ```python
            # Simplified from _col inside like() for Sym
            prior_term = the.m * prior # 'prior' calculated earlier
            count_v = col.has.get(v, 0)
            likelihood_sym = (count_v + prior_term) / (col.n + the.m + 1/BIG)
            ```
    3.  **Combine Probabilities:** Multiplies the prior probability by the likelihoods from all relevant features (`x` columns). To avoid numerical underflow with many small probabilities, it calculates the sum of the *logarithms* of the probabilities. Higher log-likelihood means higher overall probability.
        ```python
        # Inside like()
        likelihoods = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]
        log_likelihood = sum(math.log(n) for n in likelihoods + [prior] if n > 0)
        return log_likelihood
        ```
* **Smoothing (`the.k`, `the.m`):** Prevents zero probabilities if a value hasn't been seen yet in a subset. Adds a small pseudo-count.

---

## 3. Python Notes (Beyond the Basics)

`bl.py` uses some Python features that might be less familiar to beginners:

### The `o` Class and `lambda`

* The script defines a very simple class `o`:
    ```python
    class o:
      # Uses lambda for concise anonymous function definition
      __init__ = lambda i,**d: i.__dict__.update(**d)
      __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)
    ```
    * `__init__`: The constructor. `lambda i, **d: ...` defines an anonymous function that takes `self` (conventionally named `i` here) and an arbitrary dictionary of keyword arguments (`**d`). It updates the instance's internal dictionary (`__dict__`) with these arguments, effectively creating attributes on the fly. `my_obj = o(a=1, b=2)` creates an object `my_obj` with `my_obj.a == 1` and `my_obj.b == 2`.
    * `__repr__`: Defines how the object is represented as a string (e.g., when printed). It uses the helper `show` function for formatting.
    * `lambda`: Used here for extreme conciseness. A more conventional definition would be:
        ```python
        # class o:
        #  def __init__(self, **kwargs):
        #      self.__dict__.update(kwargs)
        #  def __repr__(self):
        #      return self.__class__.__name__ + show(self.__dict__)
        ```

### List/Dictionary Comprehensions

* Provide a concise way to create lists or dictionaries.
    ```python
    # List comprehension in like()
    tmp = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]

    # Dictionary comprehension in initializing 'the'
    the = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
    ```

### Generators (`yield`)

* The `csv` function uses `yield` to create a generator. Instead of reading the whole file into memory at once, it processes and `yield`s one row at a time. This is memory-efficient for large files.
    ```python
    def csv(file):
      with open(...) as src:
        for line in src:
          # ... process line ...
          if line: yield [coerce(s) for s in line.split(",")]
    ```

### `__name__ == "__main__"`

* This standard Python idiom checks if the script is being run directly (not imported as a module). Code inside this block only executes when the script is the main program.
    ```python
    if __name__ == "__main__":
      main() # Calls the main function
    ```

### Command-Line Arguments (`sys.argv`, `re`)

* `sys.argv`: A list containing the command-line arguments passed to the script. `sys.argv[0]` is the script name itself.
* The `cli` function parses these arguments:
    ```python
    def cli(d): # 'd' is the.__dict__
      for k,v in d.items(): # Iterate through default options
        for c,arg in enumerate(sys.argv): # Iterate through cmd line args
          if arg == "-"+k[0]: # Check if arg matches option flag (e.g., -f)
            # Get the value after the flag, handle boolean toggles
            new = sys.argv[c+1] if c < len(sys.argv) - 1 else str(v)
            d[k] = coerce(...) # Update the option in 'd'
    ```
* `re` (Regular Expressions): Used initially to parse the default options directly from the script's docstring.
    ```python
    regx = r"-\w+\s*(\w+).*=\s*(\S+)" # Regex to find "-option value = default"
    the = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
    ```

### File Handling (`with open`)

* Uses the `with open(...) as ...:` construct for reading files (`csv` function). This ensures the file is automatically closed even if errors occur.

---

## 4. Core Data Structures

The way data is represented is crucial.

### Columns: `Num` and `Sym`

Data is assumed to be in columns, and each column is either numerical or symbolic.

* **`Num` (Numerical Column):**
    * Represents columns with continuous numbers (integers or floats).
    * Initialized via `Num(txt="ColName", at=index)`.
    * Key Attributes:
        * `it`: Type identifier (`Num`).
        * `txt`: Column name.
        * `at`: Column index (0-based).
        * `n`: Count of non-missing values seen.
        * `mu`: Mean of values seen.
        * `sd`: Standard deviation of values seen.
        * `m2`: Sum of squares of differences from the mean (for online `sd` calculation).
        * `lo`, `hi`: Minimum and maximum values seen.
        * `goal`: Objective direction (1 for maximize `+`/`!`, 0 for minimize `-`).
    ```python
    def Num(txt=" ", at=0):
      return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG,
               goal = 0 if txt[-1]=="-" else 1)
    ```

* **`Sym` (Symbolic Column):**
    * Represents columns with discrete categories (strings or distinct values).
    * Initialized via `Sym(txt="ColName", at=index)`.
    * Key Attributes:
        * `it`: Type identifier (`Sym`).
        * `txt`: Column name.
        * `at`: Column index (0-based).
        * `n`: Count of non-missing values seen.
        * `has`: Dictionary storing counts for each category `{value: count}`.
        * `most`: Frequency (count) of the mode.
        * `mode`: The most frequent category seen.
    ```python
    def Sym(txt=" ", at=0):
      return o(it=Sym, txt=txt, at=at, n=0, has={}, most=0, mode=None)
    ```

### Column Roles: `x` vs `y`

Columns are assigned roles based on their names, influencing how they are used:

* **Naming Convention:**
    * Starts with **Uppercase** (e.g., `Weight`, `Acc`): Assumed `Num`.
    * Starts with **Lowercase** (e.g., `cylinders`, `origin`): Assumed `Sym`.
    * Ends with **`+`** or **`-`**: Objective column (`y`) to be maximized or minimized.
    * Ends with **`!`**: The single main class/objective column (`y`, `klass`).
    * Ends with **`X`**: Column to be ignored (`x`).
    * **Otherwise**: Regular independent feature column (`x`).

* **`x` Columns (Independent Features):** Used as input features, typically for calculating likelihood (`like`) or potentially distance in feature space (though distance between rows isn't explicitly used in `actLearn`'s core logic, only `ydist`).
* **`y` Columns (Dependent Objectives):** Used to evaluate how "good" a row is (`ydist`). These define the goals of the optimization.

### Managing Columns: `Cols`

* The `Cols` object is created once from the header row (list of names) and manages all the `Num` and `Sym` column objects.
* Initialized via `Cols(names)`.
* Key Attributes:
    * `it`: Type identifier (`Cols`).
    * `names`: Original list of column names.
    * `all`: A list containing all the created `Num` and `Sym` objects in order.
    * `x`: A list of the independent column objects.
    * `y`: A list of the dependent/objective column objects.
    * `klass`: Reference to the single class column object (if one ends with `!`), otherwise -1.

    ```python
    def Cols(names):
      cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
      for n,s in enumerate(names): # n=index, s=name
        # Create Num or Sym based on first char capitalization
        col = (Num if s[0].isupper() else Sym)(s,n)
        cols.all += [col]
        # Check last character for role
        if s[-1] != "X": # If not ignored
          # Add to 'y' if objective/class, otherwise to 'x'
          (cols.y if s[-1] in "+-!" else cols.x).append(col)
          if s[-1] == "!": cols.klass = col # Set the class column
      return cols
    ```

### The Dataset: `Data`

* Represents the entire dataset.
* Initialized via `Data(src=[])`, where `src` is typically the output of the `csv` generator.
* Key Attributes:
    * `it`: Type identifier (`Data`).
    * `cols`: The `Cols` object managing the columns (created from the first row of `src`).
    * `rows`: A list containing all the data rows (as lists). Each row is processed by `add`.
    * `n`: Total number of rows added.
* Also includes a `clone` method to create a new `Data` object with the same columns but potentially different rows.

    ```python
    def Data(src=[]): return adds(src, o(it=Data,n=0,rows=[],cols=None))

    # adds() handles the initialization logic:
    # If i.cols is None (first row), create i.cols = Cols(v)
    # Otherwise, add row v to i.rows and update column stats

    def clone(data, src=[]): return adds(src, Data([data.cols.names]))
    # Creates a new Data shell with the same columns, then adds rows from src
    ```

---

## 5. Efficient Updates: Incremental `add` and `sub`

A key feature for efficiency, especially in `actLearn`, is the ability to add and remove data points from summaries (`Num`, `Sym`, `Data`) without recalculating everything from scratch.

### Why Incremental?

* **Speed:** Recalculating mean, sd, or symbol counts over thousands of rows repeatedly is slow. Incremental updates take constant time (O(1)) per update, regardless of dataset size.
* **Active Learning:** `actLearn` frequently moves data points between the `best` and `rest` sets. Using `add` and `sub` makes these transfers very fast.

### How `add` Works

The `add(v, i)` function adds value `v` to summary object `i`.

* **If `i` is `Data`:**
    * If `i.cols` is not yet defined (first row `v`), creates `i.cols = Cols(v)`.
    * Otherwise, iterates through `i.cols.all`, calls `add(v[c.at], c)` for each column `c` to update its stats, and appends the raw row `v` to `i.rows`.
* **If `i` is `Num`:**
    * Increments `i.n`.
    * Updates `i.lo` and `i.hi`.
    * Updates `i.mu`, `i.m2`, and `i.sd` using the online algorithm.
* **If `i` is `Sym`:**
    * Increments `i.n`.
    * Updates the count for value `v` in `i.has`.
    * Updates `i.mode` and `i.most` if `v` becomes the new mode.

```python
def add(v, i):
  # Inner functions _data, _sym, _num contain the logic described above
  def _data(): ...
  def _sym(): ...
  def _num(): ...

  if v != "?": # Ignore missing values for statistics
    i.n += 1
    _sym() if i.it is Sym else (_num() if i.it is Num else _data())
  return v
```

### How `sub` Works

The `sub(v, i)` function reverses the `add` operation. This requires careful handling of the online statistics.

* **If `i` is `Data`:** Calls `sub(v[col.at], col)` for each column. *Note: It doesn't remove the row from `i.rows` here; that's handled by the caller, e.g., `best.rows.pop(-1)` in `actLearn`*.
* **If `i` is `Num`:**
    * Decrements `i.n`.
    * Updates `i.mu`, `i.m2`, and `i.sd` by reversing the online algorithm steps. Requires `n > 1`. If `n` becomes < 2, stats are reset.
* **If `i` is `Sym`:**
    * Decrements `i.n`.
    * Decrements the count for `v` in `i.has`. *Note: This implementation doesn't explicitly recalculate the mode if the current mode's count drops. This might be a slight simplification or potential minor inaccuracy if `sub` is used heavily without recalculation.*

```python
def sub(v, i):
   # Inner functions _data, _sym, _num contain the logic described above
   def _data(): [sub(v[col.at],col) for col in i.cols.all]
   def _sym() : i.has[v] -= 1
   def _num():
     if i.n < 2: i.mu = i.sd = 0
     else:
       # Reverse the online update steps
       d     = v - i.mu
       i.mu -= d / i.n # Note: i.n has already been decremented here
       i.m2 -= d * (v - i.mu) # Uses the *new* mu
       i.sd  = (max(0,i.m2)/(i.n-1))**.5

   if v != "?":
     i.n -= 1 # Decrement n *before* calling sub-functions for Num/Sym
     _sym() if i.it is Sym else (_num() if i.it is Num else _data())
   return v
```

---

## 6. AI Tools (Simple to Complex)

These functions use the data structures and core operations to perform the optimization and learning.

### Evaluating Rows: `ydist`

* **Purpose:** Quantifies how "good" a single row is according to the defined objectives (`y` columns). Lower distance is better.
* **How:**
    1.  Iterates through the objective columns (`data.cols.y`).
    2.  For each column `c`, normalizes the row's value (`norm(row[c.at], c)`).
    3.  Calculates the absolute difference between the normalized value and the column's goal (`c.goal`, which is 0 for `-` cols, 1 for `+`/`!` cols).
    4.  Raises this difference to the power `the.p`.
    5.  Sums these powered differences across all `y` columns.
    6.  Divides by the number of `y` columns (average powered distance).
    7.  Takes the `1/the.p` root to get the final distance.

    ```python
    def ydist(row,  data):
      return (sum(abs(norm(row[c.at], c) - c.goal)**the.p for c in data.cols.y)
              / len(data.cols.y)) ** (1/the.p)

    # Helper to sort rows by their ydist
    def ydists(rows, data): return sorted(rows, key=lambda row: ydist(row,data))

    # Helper to get summary stats of ydist values for a set of rows
    def yNums(rows,data): return adds(ydist(row,data) for row in rows)
    ```

### Calculating Likelihood: `like` and `likes`

* **Purpose:** Estimate how likely a `row` is to belong to a given `Data` subset (cluster). Used by `actLearn` to decide where a candidate point fits best.
* **`like(row, data, nall, nh)`:**
    * Calculates the log-likelihood of the `row` belonging to the `data` subset.
    * `nall`: Total number of examples across all compared subsets.
    * `nh`: Number of subsets being compared (hyper-parameter, typically 2 for best/rest).
    * Steps:
        1.  Calculate the `prior` probability of the `data` subset (smoothed).
        2.  Initialize a list of likelihoods with the `prior`.
        3.  Iterate through the independent columns (`data.cols.x`):
            * Get the value `v = row[x.at]`.
            * If `v` is not missing (`?`):
                * Call `_col(v, x)` to get the likelihood of `v` given column `x`'s statistics in `data` (using Gaussian PDF for `Num`, smoothed frequency for `Sym`).
                * Append this likelihood to the list.
        4.  Sum the natural logarithms (`math.log`) of all non-zero likelihoods in the list.
        5.  Return the total log-likelihood.

    ```python
    def like(row, data, nall=100, nh=2):
      def _col(v,col): # Calculates likelihood for one column
        if col.it is Sym:
          # Smoothed frequency P(v|SymCol)
          return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
        # Gaussian PDF P(v|NumCol)
        sd    = col.sd + 1/BIG
        nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
        denom = (2*math.pi*sd*sd) ** 0.5
        return max(0, min(1, nom/denom)) # Clamp probability

      prior = (data.n + the.k) / (nall + the.k*nh) # Smoothed prior P(data)
      # Get likelihoods P(xi|data) for all features xi
      tmp   = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]
      # Return Sum(log(P(xi|data))) + log(P(data))
      return sum(math.log(n) for n in tmp + [prior] if n>0)
    ```
* **`likes(lst, datas)`:**
    * Given a row (`lst`) and a list of `Data` objects (`datas`), finds the `Data` object where the row has the highest `like` score.

    ```python
    def likes(lst, datas):
      n = sum(data.n for data in datas) # Total examples across datas
      # Return data object that maximizes like(lst, data, ...)
      return max(datas, key=lambda data: like(lst, data, n, len(datas)))
    ```

### Intelligent Data Selection: `actLearn`

* **Purpose:** The main active learning algorithm. Selects a small subset of rows (`best.rows`) that are likely to be good performers according to `ydist`, by iteratively querying the most informative points.
* **How:**
    1.  **Initialization:**
        * Takes the full `data` as input.
        * Selects an initial small set of rows (`data.rows[:the.start]`).
        * Sorts these initial rows by `ydist` (`ydists`).
        * Splits them into `best` (the top `n**the.guess` rows) and `rest`. `the.guess` controls the initial split ratio.
        * Creates `Data` clones `best` and `rest`, populated with these rows and their stats (using `clone` and `add`).
        * The remaining rows form the pool of candidates (`todo`).
    2.  **Iteration Loop:** Continues while there are candidates (`len(todo) > 2`) and the evaluation limit (`the.Stop`) hasn't been reached.
        * **Candidate Selection:**
            * Takes a batch of candidates from the front of `todo` (size `the.Guesses`).
            * For each candidate `row` in the batch, calculate a score using `_guess(row)`.
                * `_guess(row)` calls `_acquire(p, b, r)` where:
                    * `p` is progress `n/the.Stop`.
                    * `b` is `like(row, best, ...)`.
                    * `r` is `like(row, rest, ...)`.
                * `_acquire` calculates the acquisition score based on the chosen strategy (`the.acq`):
                    * `xploit`: Favors points very likely to belong to `best` (`b`).
                    * `xplore`: Favors points where `best` vs `rest` likelihood is uncertain or favors `rest` (`r`).
                    * `adapt`: Starts like `xplore` and shifts towards `xploit` as progress `p` increases.
                    * The formula `(b + r*q) / abs(b*q - r + 1/BIG)` combines these likelihoods based on the exploration factor `q`.
            * Sorts the batch by this acquisition score (descending - higher score is better).
            * Selects the top-scoring candidate (`top`).
        * **Update `todo`:** The script shuffles `todo` slightly by moving the non-selected candidates from the batch (`others`) around. This adds some randomness.
        * **Update `best` and `rest`:**
            * Adds the selected `top` row to the `best` set using `add`.
            * Re-sorts `best.rows` by `ydist`.
            * **Balancing:** If `best` grows too large (heuristically `> n**0.5`), the *worst* row (highest `ydist`) is removed from `best` (using `sub`) and added to `rest` (using `add`). This keeps `best` focused on top performers.
        * Increment `n`.
    3.  **Return:** Returns the final `best.rows`.

    ```python
    def actLearn(data):
      def _guess(row): # Score a candidate row
        # Calculate likelihoods for best and rest
        b_like = like(row, best, n, 2)
        r_like = like(row, rest, n, 2)
        # Call acquisition function
        return _acquire(n/the.Stop, b_like, r_like)

      def _acquire(p, b,r): # Acquisition function
        b,r = math.e**b, math.e**r # Convert log-likelihoods back to probabilities
        # Determine exploration factor 'q' based on strategy
        q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
        # Combine likelihoods based on 'q'
        return (b + r*q) / abs(b*q - r + 1/BIG) # Score emphasizes b if q=0, balances if q=1

      # --- Initialization ---
      n     = the.start
      todo  = data.rows[n:] # Candidate pool
      done  = ydists(data.rows[:n], data) # Initial evaluated set
      cut   = round(n**the.guess) # Split point for best/rest
      best  = clone(data, done[:cut]) # Clone structure, add best rows
      rest  = clone(data, done[cut:]) # Clone structure, add rest rows

      # --- Iteration Loop ---
      while len(todo) > 2  and n < the.Stop:
        n += 1
        # Select candidates, score with _guess, pick top one
        top, *others = sorted(todo[:the.Guesses], key=_guess, reverse=True)

        # Re-arrange todo slightly (adds some stochasticity)
        m = int(len(others)/2)
        todo = others[:m] + todo[the.Guesses:] + others[m:]

        # Add top row to best set
        add(top, best)
        best.rows = ydists(best.rows, data) # Re-sort best

        # If best is too big, move worst from best to rest
        if len(best.rows) > n**0.5: # Heuristic size limit
            worst_row = best.rows.pop(-1) # Remove worst row (highest ydist)
            add( sub(worst_row, best), rest) # Update stats via sub/add

      return best.rows # Return the final identified best rows
    ```

---

## 7. Appendix: Configuration Options (`the`)

These options control the script's behavior and are set via command-line flags (e.g., `-f newfile.csv -s 10`). They are stored in the `the` object.

* **`-a acq`** (`the.acq`)
    * **Purpose:** Active learning acquisition strategy.
    * **Options:** `xploit` (exploitation - refine known good areas), `xplore` (exploration - probe uncertain areas), `adapt` (start exploring, shift to exploiting).
    * **Default:** `xploit`

* **`-d decs`** (`the.decs`)
    * **Purpose:** Number of decimal places for printing numeric output.
    * **Default:** `3`

* **`-f file`** (`the.file`)
    * **Purpose:** Path to the input training CSV file.
    * **Default:** `../test/data/auto93.csv`

* **`-g guess`** (`the.guess`)
    * **Purpose:** Exponent controlling the initial split size between `best` and `rest` (`best` size = `start**guess`). Smaller value means smaller initial `best` set.
    * **Default:** `0.5`

* **`-G Guesses`** (`the.Guesses`)
    * **Purpose:** Number of candidate rows evaluated in each iteration of `actLearn` to select the next point.
    * **Default:** `100`

* **`-k k`** (`the.k`)
    * **Purpose:** Laplace smoothing parameter for calculating the prior probability in `like`.
    * **Default:** `1`

* **`-l leaf`** (`the.leaf`)
    * **Purpose:** Minimum size of leaves in a tree (Seems related to the unused discretization/merge code).
    * **Default:** `2`

* **`-m m`** (`the.m`)
    * **Purpose:** Laplace smoothing parameter for calculating symbolic likelihoods in `like`.
    * **Default:** `2`

* **`-p p`** (`the.p`)
    * **Purpose:** Exponent for the Minkowski distance calculation in `ydist`. `p=1` is Manhattan, `p=2` is Euclidean.
    * **Default:** `2`

* **`-r rseed`** (`the.rseed`)
    * **Purpose:** Random number seed for reproducibility.
    * **Default:** `1234567891`

* **`-s start`** (`the.start`)
    * **Purpose:** Initial number of rows to evaluate before starting the active learning loop.
    * **Default:** `4`

* **`-S Stop`** (`the.Stop`)
    * **Purpose:** Maximum number of rows to evaluate in total during active learning (`start` + iterations). Controls the budget.
    * **Default:** `32`
