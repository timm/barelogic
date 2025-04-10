#!/bin/bash

make acts1 | gawk '{print NR,$13,$4, $16, $19,$22, $25,$28,$31}'> ~/tmp/all.data

gnuplot<<EOF
set output "~/tmp/all.png"
set style data linespoints
set datafile separator whitespace
set xlabel "results from 56 data sets, 20 trials (sorted by n=200)"
set ylabel "mean distance to heaven"
set ytics 0.1
set yrange [-0.01:0.7]
set xrange [1:56]

set grid linetype 0
set grid linewidth 0.5
set grid linecolor "grey"
set terminal pngcairo font 'Arial,15'
set key top left
set terminal png size 1000,400
set key outside;
set key right top;
plot \
  '~/tmp/all.data' using 1:2 title 'b4.mu' with lines lw 3 linecolor rgb "black", \
  '~/tmp/all.data' using 1:3 title 'b4.lo' with lines lw 3, \
  '~/tmp/all.data' using 1:9 title 'n=8' with lines, \
  '~/tmp/all.data' using 1:8 title 'n=16' with lines, \
  '~/tmp/all.data' using 1:7 title 'n=32' with lines lw 3, \
  '~/tmp/all.data' using 1:6 title 'n=64' with lines, \
  '~/tmp/all.data' using 1:5 title 'n=128' with line, \
  '~/tmp/all.data' using 1:4 title 'n=256' with line
EOF

#cat ~/tmp/$1/*.csv | gawk  '{print $16}' > ~/tmp/${1}s
