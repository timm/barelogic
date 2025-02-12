-include ../src/Makefile

p = export PYTHONPATH="..:$$PYTHONPATH" 
q = $p; python3 -B

c = $p; coverage run $@.py; coverage report
r = gawk '/bl.py/ {sub(/[^[0-9\.]/,"",$$4); print $$4}'

all: lib data nb kmeans

nb     :; $c | $r
lib    :; $c | $r
data   :; $c | $r
load   :; $c | $r
kmeans :; $c | $r
