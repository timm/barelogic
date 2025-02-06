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

push: ## commit to main
	- echo -en "$(LOUD)Why this push? $(SOFT)" ;  read x ; git commit -am "$$x" ;  git push
	- git status

sh: ## run my shell
	bash --init-file  ../etc/dotshellrc -i


../docs/%.html : %.py
	pycco -d ../docs  $^
	echo "p { text-align:right; }" >> ../docs/pycco.css
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


