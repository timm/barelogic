<h1> Bare Logic </h1>
<h3>Think Fast. Learn Smart. Stay Wild. <br>(Case studies in simpler AI,  elegance, and the wisdom of reduction).</h3>
<p>
<img src="/docs/img/barelogic.png"  width="250">
<br>
<a href="http://github.com/timm/barelogic"><img src="https://img.shields.io/badge/GitHub-src-yellow?logo=github&style=flat-square"></a> 
<img alt="language" src="https://img.shields.io/badge/language-python-blue.svg?logo=python&logoColor=white&style=flat-square"> 
<a href="https://github.com/timm/barelogic/blob/main/LICENSE.md"><img alt="license" src="https://img.shields.io/badge/license-MIT-brightgreen?logo=open-source-initiative&logoColor=white&style=flat-square"></a>
<a href="https://github.com/timm/barelogic/blob/main/README.md"><img alt="docs" src="https://img.shields.io/badge/docs-available-orange?logo=readthedocs&logoColor=white&style=flat-square"></a>
<a href="http://github.com/timm/barelogic/issues"><img alt="issues" src="https://img.shields.io/badge/issues-track-red?logo=github&style=flat-square"></a>
<a href="src/Makefile"><img src="/docs/img/pylint.svg"></a>

</p>


Why BareLogic? Why reinvent a minimal version of useful AI tools?

Well, lets ask 
[Richard Hipp](https://corecursive.com/066-sqlite-with-richard-hipp/)
the author SQLite, 
who is famous for (re)building the tools he  uses day to day.

<img src="/docs/img/backpack.png" align=right width=300> 

 >  "People go backpacking,
 long hikes, where they carry
 everything they need on their back and they talk about how freeing
 that is because they are taking care of themselves."


> "You’re not dependent on different vendors providing you something. Suppose I had elected to go with Berkeley DB as the storage engine for SQLite version two.
Well, at that time Berkeley DB was open source, but then later it was sold off to Oracle, and it became a duel source proprietary model and you can’t get hold of the source code to the later versions without paying license fees, yada, yada, yada, so, suddenly it becomes a problem."



>  "... (Pack backing)  **involves doing without**. ....
Whenever a politician comes to you and says, “Oh, we’re
going to take care of this problem for you.” What they’re really
saying is, we’re going to take away some of your freedoms. The
exercise for you, here, is to figure out what freedoms they’re going
to take away in order to solve this problem, and they’re often
well-disguised, but yeah, if you want to be free, that means doing
things yourself."

It turns out that "doing without" is an important principles:

- William of Occum warned that
 _"Frustra fit per plura quod potest fieri per pauciora"_  (It is futile to do with more things that which can be done with fewer).
- Aristotle told us _"Ἡ φύσις τῇ ἐλαχίστῃ ὁδῷ ἐργάζεται"_ (Nature operates in the shortest way possible). And the maths supports him.
- Consider the space of systems all around us that have stood the test of time.
  Mathematically, it is more like that those surviving things are less complex.
  Nature optimizes towards some goal (e.g.  Survival) and that means 
  it must test  many different ways of doing things. 
  Simpler ways (with less decisions) are more likely to 
      apply to more future similar problems
      (because "similarity" means checking for fewer things) [^apply].
  Also, simpler things are easier to test if, for no other reason,
      it will be easier to find or make repeated similar events (again, since the fewer the
      choices, the easier it to find something like those choices)
  On the other hand, more complex ways (that rely on more decisions ) struggle to
  generalize because any two decisions, even those from some same class, tend to be further  apart [^dist].

[^apply]: In Euclidean geometry, decisions partition space into
regions. Simpler methods, with fewer decisions, create larger
regions, covering more cases (past and present and future). 

[^dist]: As dimensions increase, distances between points **cannot decrease** and generally **increase**;
i.e. items are further apart in higher dimensions. 
Consider two points in an n-dimensional space:
     `A = (x₁, x₂, ..., xₙ)` and  `B = (y₁, y₂, ..., yₙ)`
The Euclidean distance between them is
`dₙ = sqrt((x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)²)`.
If we extend these points into (n+1)-dimensional space by adding new coordinates xₙ₊₁ and yₙ₊₁, the distance becomes:
`dₙ₊₁ = sqrt((x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)² + (xₙ₊₁ - yₙ₊₁)²)`.
Since (xₙ₊₁ - yₙ₊₁)² is always non-negative, we get:
`dₙ₊₁ ≥ dₙ`.  Q.E.D.



Is all that too long-winded for you? Well how about:

<img src="/docs/img/song.png" align=right width=300> 

> _"La perfection est atteinte, non pas lorsqu’il n’y a plus rien à ajouter, mais lorsqu’il n’y a plus rien à enlever."_
(Perfection is achieved, not when there is nothing more to add, but when there is nothing 
left to take away.)<br>
― Antoine de Saint-Exupéry, Airman's Odyssey


"Taking away" things is another important
principle. In his book [Empirical Methods for AI](https://www.eecs.harvard.edu/cs286r/courses/spring08/reading6/CohenTutorial.pdf), 
William Cohen argues that
supposedly sophisticated methods should be benchmarked
against seemingly stupider ones (the so-called “straw man”
approach). I can attest that when ever I checked a supposedly sophisticated method against
a simpler one, there was always something useful in the
simpler. More often than not, a year later, I had
switched to the simpler approach.

So take every you know about AI. Then see what you can "do without"
and what can you can "take away".
Repeat that for decades of AI programming,
 and will be on your way to building  your  own BareLogic.

And if you can't wait for decades, you could just look over my stuff.
Share and enjoy [^hhgth].


[^hhgth]: From the Hitchhikers Guide to the Galaxy by Douglas Adams. “'Share and Enjoy' is the company motto of the hugely successful Sirius Cybernetics Corporation Complaints Division, which now covers the major land masses of three medium-sized planets and is the only part of the Corporation to have shown a consistent profit in recent years. The motto stands-- or rather stood-- in three mile high illuminated letters near the Complaints Department spaceport on Eadrax. Unfortunately its weight was such that shortly after it was erected, the ground beneath the letters caved in and they dropped for nearly half their length through the offices of many talented young Complaints executives-- now deceased. The protruding upper halves of the letters now appear, in the local language, to read 'Go stick your head in a pig,' and are no longer illuminated, except at times of special celebration.”

