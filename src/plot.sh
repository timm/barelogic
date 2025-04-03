#!/bin/bash

cat $1/* |
gawk '
  BEGIN { OFS = " " }
  { mu =  new = sd =  ""
    for (i = 1; i <= NF; i++) {
      if ($i == ":mu")   mu   = $(i+1)
      if ($i == ":new")  new  = $(i+1)
      if ($i == ":sd") sd = $(i+1)}
    if (mu != "") { print mu, new, sd }}
' | sort -n  -k 2 | cat -n > $1/data

gnuplot<<EOF
set terminal png size 1000,300
set output "$1/data.png"
set key outside
set style data linespoints
set datafile separator whitespace
set xlabel "results from 20 trials. Sorted by new.mu"
set ylabel "d"
set ytics 0.2
set yrange [-0.05:0.8]
plot \\
  '$1/data' using 1:2 title 'b4.mu' with lines, \\
  '$1/data' using 1:3:4 title 'new.mu ± sd' with yerrorlines
EOF
