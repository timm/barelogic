#!/bin/bash

cat ~/tmp/$1/*.csv |
gawk '
  BEGIN { OFS = " " }
  { mu =  new = sd =  ""
    for (i = 1; i <= NF; i++) {
      if ($i == ":mu0")   mu   = $(i+1)
      if ($i == ":mu1")  new  = $(i+1)
      if ($i == ":sd1") sd = $(i+1)}
    if (mu != "") { print mu, new, sd }}
' | sort -n  -k 2 | cat -n > ~/tmp/$1/$1

gnuplot<<EOF
set terminal png size 1000,300
set output "~/tmp/$1/$1.png"
set key outside
set style data linespoints
set datafile separator whitespace
set xlabel "results from 20 trials. Sorted by new.mu"
set ylabel "d"
set ytics 0.1
set yrange [-0.05:0.8]

set grid linetype 0
set grid linewidth 0.5
set grid linecolor "grey"

plot \\
  '~/tmp/$1/$1' using 1:2 title 'asIs.mu' with lines, \\
  '~/tmp/$1/$1' using 1:3:4 title 'toBe.mu ± sd' with yerrorlines
EOF

cat ~/tmp/$1/*.csv | gawk  '{print $16}' > ~/tmp/${1}s
