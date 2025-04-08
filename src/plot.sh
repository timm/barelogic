#!/bin/bash

rm /Users/timm/tmp/*.dat

for i in eg6 eg12 eg24 eg50 eg100 eg200; do
	cat ~/tmp/$i/*.csv |
	gawk '
	  BEGIN { OFS = " " }
	  { mu0 = mu1 =  ""
	    for (i = 1; i <= NF; i++) {
	      if ($i == ":mu0")   mu0   = $(i+1)
	      if ($i == ":mu1")  mu1  = $(i+1) }
	    if (mu0 != "") { print mu0,mu1 }}
	'  > ~/tmp/$i.dat
  echo ~/tmp/$i.dat
done 

paste           \
  ~/tmp/eg6.dat  \
  ~/tmp/eg12.dat  \
  ~/tmp/eg24.dat   \
  ~/tmp/eg50.dat    \
  ~/tmp/eg100.dat    \
  ~/tmp/eg200.dat  | sort -n -k 12 | cat -n > ~/tmp/all.data

gawk '{z+=$2; a+=$3; b+=$5; c+=$7; d+=$9; e+=$11; f+= $13}
END  { printf("%.1f  %.1f %.1f %.1f %.1f %.1f %.1f",z,  a, b, c, d, e, f) }
' ~/tmp/all.data

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
set terminal png size 1000,300
set key top left
plot \
  '~/tmp/all.data' using 1:2 title 'asIs' with lines lw 2, \
  '~/tmp/all.data' using 1:3 title 'n=6' with lines, \
  '~/tmp/all.data' using 1:5 title 'n=12' with lines, \
  '~/tmp/all.data' using 1:7 title 'n=24' with lines, \
  '~/tmp/all.data' using 1:9 title 'n=50' with lines, \
  '~/tmp/all.data' using 1:11 title 'n=100' with lines, \
  '~/tmp/all.data' using 1:13 title 'n=200' with lines #yerrorlines
EOF

#cat ~/tmp/$1/*.csv | gawk  '{print $16}' > ~/tmp/${1}s
