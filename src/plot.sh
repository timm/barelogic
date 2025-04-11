#!/bin/bash

make acts1 | grep -v '#' | gawk '{print NR,$7,$2, $9, $11,$13, $15,$17,$19}'> ~/tmp/all.data

gawk '{a+=$2; b+=$3; c+=$4; d+=$5;e+=$6;f+=$7; g+=$8;h+=$9}  \
  function n(x) {return 100-int(100*(x-b)/(a-b)) }
END {print n(a), n(b), n(c), n(d), n(e), n(f), n(g), n(h)}' ~/tmp/all.data

# returning top
cat<<EOF> /tmp/plot.data
8   63
16  77
32  84
64  89
128 94
256 96
EOF

# # return all
# cat <<EOF > /tmp/plot.data
# 8 56
# 16 54 
# 32 74 
# 64 72 
# 128 82 
# 256 80 
# EOF

gnuplot <<'EOF'
set terminal pngcairo size 1000,400 enhanced font 'Arial,10'
set output '~/tmp/combined.png'
set datafile separator whitespace

set multiplot

# --- Main plot (left 72%) ---
set origin 0.0, 0.0
set size 0.72, 1.0

set key right bottom
set style data linespoints
set xlabel "means seen in 20 trials, data sets sorted by b4.lo"
set ylabel "mean distance to heaven"
set ytics 0.1
set yrange [0:0.75]

set grid ytics
set grid linetype 0 linewidth 0.5 linecolor "grey"

plot \
  '~/tmp/all.data' using 1:2 title 'b4.mu' with lines lw 3 linecolor rgb "black", \
  '~/tmp/all.data' using 1:3 title 'b4.lo' with lines lw 3, \
  '~/tmp/all.data' using 1:9 title 'n=8' with lines, \
  '~/tmp/all.data' using 1:8 title 'n=16' with lines, \
  '~/tmp/all.data' using 1:7 title 'n=32' with lines lw 3, \
  '~/tmp/all.data' using 1:6 title 'n=64' with lines, \
  '~/tmp/all.data' using 1:5 title 'n=128' with lines, \
  '~/tmp/all.data' using 1:4 title 'n=256' with lines

# --- Histogram (right 28%) ---
set origin 0.72, 0.0
set size 0.28, 1.0

unset title
unset key
set xlabel "number of samples"
set ylabel "%max optimization"

unset style data
set style fill solid 1.0 border 0 #1
set boxwidth 0.66
set ytics 10
set yrange [0:100]

unset xtics
set xtics rotate by -45 offset 0,-0.5
set xtics ("8" 0.5, "16" 1.5, "32" 2.5, "64" 3.5, "128" 4.5, "256" 5.5)

plot '/tmp/plot.data' using ($0 + 0.5):2 with boxes  lc rgb "gray"

unset multiplot
EOF

