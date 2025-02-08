<h1> Bare Logic </h1>
<h3> Case studies in simpler AI,  elegance, and the wisdom of reduction. Think Fast. Learn Smart. Stay Wild.</h3>
<p>
<img src="/docs/img/barelogic.png"  width="250">
<br>
<a href="http://github.com/timm/barelogic"><img src="https://img.shields.io/badge/GitHub-src-yellow?logo=github&style=flat-square"></a> 
<img alt="language" src="https://img.shields.io/badge/language-python-blue.svg?logo=python&logoColor=white&style=flat-square"> 
<a href="https://github.com/timm/barelogic/blob/main/LICENSE.md"><img alt="license" src="https://img.shields.io/badge/license-MIT-brightgreen?logo=open-source-initiative&logoColor=white&style=flat-square"></a>
<a href="https://github.com/timm/barelogic/blob/main/README.md"><img alt="docs" src="https://img.shields.io/badge/docs-available-orange?logo=readthedocs&logoColor=white&style=flat-square"></a>
<a href="http://github.com/timm/barelogic/issues"><img alt="issues" src="https://img.shields.io/badge/issues-track-red?logo=github&style=flat-square"></a>
</p>


Why BareLogic? Why reivent a minimal version of useful AI tools?

Well, lets ask 
[Richard Hipp](https://corecursive.com/066-sqlite-with-richard-hipp/)
the author SQLite, 
who is famous for (re)building the tools he  uses day to day.

<img src="/docs/img/backpack.png" align=right width=300> 

 >  People go backpacking,
 long hikes, where they carry
 everything they need on their back and they talk about how freeing
 that is because they are taking care of themselves.


> You’re not dependent on different vendors providing you something. Suppose I had elected to go with Berkeley DB as the storage engine for SQLite version two.
Well, at that time Berkeley DB was open source, but then later it was sold off to Oracle, and it became a duel source proprietary model and you can’t get hold of the source code to the later versions without paying license fees, yada, yada, yada, so, suddenly it becomes a problem.



>  ... (Packbacking)  **involves doing without**. ....
Whenever a politician comes to you and says, “Oh, we’re
going to take care of this problem for you.” What they’re really
saying is, we’re going to take away some of your freedoms. The
exercise for you, here, is to figure out what freedoms they’re going
to take away in order to solve this problem, and they’re often
well-disguised, but yeah, if you want to be free, that means doing
things yourself.

It turns out that "doing without" is an important principles:

- William of Occum warned that
 _"Frustra fit per plura quod potest fieri per pauciora"_  (It is futile to do with more things that which can be done with fewer).
- Aristotle told us _"Ἡ φύσις τῇ ἐλαχίστῃ ὁδῷ ἐργάζεται"_ (Nature operates in the shortest way possible). And the maths suports him.
 Nature optimize towards some goal (e.g. surival) and that means it must try  many different ways of doing things.
  Mathematically, simpler ways (with less decision) are easier to test and apply to future similar problems
   since they allow for more repeated events (because "similarity" means checker for fewer things).
And more complex ways (that rely on more decisions ) struggle to generalize because any two decisions, even those from some same class, tend to be further  apart.

<img src="/docs/img/song.png" align=right width=300> 

Is all that too long-winded for you? Well how about:


> La perfection est atteinte, non pas lorsqu’il n’y a plus rien à ajouter, mais lorsqu’il n’y a plus rien à enlever
(Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away). <br>
― Antoine de Saint-Exupéry, Airman's Odyssey


In his book [Empirical Methods for AI](https://www.eecs.harvard.edu/cs286r/courses/spring08/reading6/CohenTutorial.pdf), 
William Cohen argues that
supposedly sophisticated methods should be benchmarked
against seemingly stupider ones (the so-called “straw man”
approach). I can attest that when ever I checked a supposedly sophisticated method against
a simpler one, there was always something useful in the
simpler. More often than not, a year later, I have
switched to the simpler approach.

And if you repeat that "switch to the simpler"  stuff for 40 years of AI programming, you get BareLogic.
