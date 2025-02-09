-include ../src/Makefile

p=export PYTHONPATH="..:$$PYTHONPATH"; python3 -B

all: lib data nb kmeans

lib:
	$p lib.py

data:
	$p data.py

nb:
	$p nb.py

kmeans:
	$p kmeans.py 
