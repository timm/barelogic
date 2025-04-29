# Default is show help; e.g.
#
#    make 
#
# prints the help text.

SHELL     := bash
MAKEFLAGS += --warn-undefined-variables
.SILENT:

LOUD = \033[1;34m#
HIGH = \033[1;33m#
SOFT = \033[0m#

Top=$(shell git rev-parse --show-toplevel)
Tmp  ?= $(HOME)/tmp

help      :  ## show help
	gawk -f $(Top)/etc/help.awk $(MAKEFILE_LIST) 

pull: ## update from main
	git pull

push: ## commit to main
	- echo -en "$(LOUD)Why this push? $(SOFT)" 
	- read x ; git commit -am "$$x" ;  git push
	- git status

sh: ## run my shell
	bash --init-file  $(Top)/etc/dotshellrc -i

luac:
	luacheck bll.lua --ignore 61*
lint: ## lint all python in this directory
	export PYTHONPATH="..:$$PYTHONPATH"; \
	pylint --disable=W0311,C0303,C0116,C0321,C0103 \
		    --disable=C0410,C0115,C3001,R0903,E1101 \
		    --disable=E701,W0108,W0106,W0718,W0201   bl.py

../docs/%.html : %.py
	pycco -d $(Top)/docs  $^
	echo "pre { font-size: small;} p { text-align:right; }" >> $(Top)/docs/pycco.css
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

../docs/%.html : %.lua
	pycco -d $(Top)/docs  $^
	echo "pre { font-size: small;} p { text-align:right; }" >> $(Top)/docs/pycco.css
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

~/tmp/%.pdf: %.py  ## make doco: .py ==> .pdf
	mkdir -p ~/tmp
	echo "pdf-ing $@ ... "
	a2ps                 \
		-Br                 \
		--chars-per-line=90 \
		--file-align=fill      \
		--line-numbers=1        \
		--pro=color               \
		--left-title=""            \
		--borders=no             \
	    --left-footer="$<  "               \
	    --right-footer="page %s. of %s#"               \
		--columns 3                 \
		-M letter                     \
	  -o	 $@.ps $<
	ps2pdf $@.ps $@; rm $@.ps
	open $@

~/tmp/%.pdf : %.lua  Makefile
	@echo "pdfing : $@ ... "
	@a2ps -Bj --landscape                           \
		--chars-per-line=90 \
		--line-numbers=1                    \
		--highlight-level=normal  \
		--columns 3                 \
		--borders=no --pro=color \
		--right-footer="" --left-footer=""    \
		--pretty-print=../etc/lua.ssh             \
		--footer="page %p."                     \
		-M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps
	open $@

fast1:
	cat ~/tmp/fast_*.out | column -t | grep --color a

#fast: ../../moot/optimize/[bchmp]*/*.csv
fast: ../../moot/optimize/[b]*/*.csv
	$(foreach d, $^, (python3 -B bl.py --fast $d | tee ~/tmp/$@_$(notdir $d).out &); )

acts1:
	cat ~/tmp/acts_*.out | sort | uniq | sed "s/[{}']//g" | sort -n -k 2  | column -s, -t | grep --color a

acts: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, (python3 -B bl.py --acts $d | tee ~/tmp/$@_$(notdir $d).out &); )

cuts: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, python3 -B bl.py --cuts $d;)

rules: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, python3 -B bl.py -l 5 --rules $d | column -t;)

afters: ../../moot/optimize/[bchmp]*/*.csv
	#$(foreach d, $^, (python3 -B bl.py -l 2 -S 200 --after $d &);)
	#$(foreach d, $^, (python3 -B bl.py -l 2 -S 100 --after $d &);)
	$(foreach d, $^, (python3 -B bl.py -l 2 -S 50 --after $d &);)
	$(foreach d, $^, (python3 -B bl.py -l 2 -S 40 --after $d &);)
	$(foreach d, $^, (python3 -B bl.py -l 2 -S 30 --after $d &);)
	$(foreach d, $^, (python3 -B bl.py -l 2 -S 20 --after $d &);)

aftersAll:
	#$(MAKE) afters | tee ~/tmp/$@.out
	$(foreach d,200 100 50 40 30 20,\
	   gawk '$$4==$d {x[NR]=$$2} END {n=int(0.1*asort(x)); print $d, x[n], x[3*n], x[5*n], x[7*n],x[9*n]}' ~/tmp/$@.out ;)

#l=2
# S   10 30 50  70 90 <== percentile
# === == == ==  == ==
# 200 21 78 93  98 100
# 100 72 89 97 100 100
#  50 66 93 97 100 100
#  40 65 85 97 100 100
#  30 65 83 93 100 100
#  20 70 89 97 100 100

# ▶ py bl.py -l 2 -S 30 --after ../../moot/optimize/process/nasa93dem.csv
# o{:win 97 :samples 30 :mu1 0.54 :mu2 0.525 :lo1 0.387 :lo2 0.391 :file "nasa93dem.csv"}


# py bl.py -l 2 --cuts ../../moot/optimize/process/nasa93dem.csv
# o{:mu1 0.54 :mu2 0.437 :sd1 0.051 :sd2 0.019}
# ▶ py bl.py -l 2 -S 30 --cuts ../../moot/optimize/process/nasa93dem.csv
# o{:mu1 0.54 :mu2 0.46 :sd1 0.051 :sd2 0.043}
# [30] 0.531
# |  rely == l [2] 0.438
# |  rely == h [10] 0.52
# |  |  data == h [2] 0.481
# |  |  data == n [6] 0.532
# |  |  |  pmat == l [3] 0.504
# |  |  |  |  cplx == vh [2] 0.506
# |  |  |  pmat == n [2] 0.569
# |  rely == n [14] 0.536
# |  |  data == n [5] 0.521
# |  |  |  pmat == l [2] 0.527
# |  |  |  pmat == n [2] 0.54
# |  |  data == l [7] 0.541
# |  |  |  pcap == h [5] 0.54
# |  |  |  |  cplx == n [3] 0.526
# |  |  |  |  cplx == h [2] 0.562
# |  |  |  pcap == vh [2] 0.543
# |  rely == vh [4] 0.587
# |  |  pmat == h [2] 0.561
# |  |  pmat == l [2] 0.613
