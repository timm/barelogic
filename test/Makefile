-include ../src/Makefile

p=export PYTHONPATH="..:$$PYTHONPATH"; python3 -B
	
tests:
	$p lib.py
	$p data.py
	$p nb.py
	$p kmeans.py | sort -n -k 3
