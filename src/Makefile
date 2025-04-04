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
		    --disable=E701,W0108,W0106,W0718   bl.py

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

eg?=eg24
pretties:
	$(MAKE) eg=eg6 pretty
	$(MAKE) eg=eg12 pretty
	$(MAKE) eg=eg24 pretty
	$(MAKE) eg=eg50 pretty
	$(MAKE) eg=eg100 pretty
	$(MAKE) eg=eg200 pretty

pretty:
	mkdir -p ~/tmp/$(eg)
	rm -f  ~/tmp/$(eg)/*
	time $(MAKE) $(eg) | sort -n -k 2 | column -t
	bash plot.sh $(eg)
	open ~/tmp/$(eg)/$(eg).png

eg6: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 6 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg12: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 12 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg18: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 18 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg24: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 24 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg50: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 50 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg100: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 100 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

eg200: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d,$^, (python3 bl.py -S 200 --actLearn $d |tee  ~/tmp/$@/$(notdir $d) &); )

