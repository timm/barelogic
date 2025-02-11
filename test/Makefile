-include ../src/Makefile

p=export PYTHONPATH="..:$$PYTHONPATH"; python3 -B

all: lib data nb kmeans

load:
		$p load.py
:
lib:
	$p lib.py

data:
	$p data.py

nb:
	$p nb.py

kmeans:
	$p kmeans.py 
