#!/bin/bash

make acts1 | grep -v '#' | gawk '{print NR,$7,$2, $9, $11,$13, $15,$17,$19}'> ~/tmp/all.data

gawk '{a+=$2; b+=$3; c+=$4; d+=$5;e+=$6;f+=$7; g+=$8;h+=$9} END {print a-b,b-b,c-b,d-b,e-b,f-b,g-b,h-b}' ~/tmp/all.data

gnuplot<<EOF
set output "~/tmp/all.png"
set style data linespoints
set datafile separator whitespace
set xlabel "means seen in 20 trials, data sets sorted by b4.lo"
set ylabel "mean distance to heaven"
set ytics 0.1
set yrange [0:0.75]
set xrange [1:]

set grid linetype 0
set grid linewidth 0.5
set grid linecolor "grey"
set terminal pngcairo font 'Arial,15'
set terminal png size 1000,400
#set key outside;
set key right bottom;
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
