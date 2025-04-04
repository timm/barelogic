#file	_6	_12	_24	_50	_100	_200
#            2           3          4             5           6         7        8
paste ~/tmp/eg200s ~/tmp/eg100s ~/tmp/eg50s ~/tmp/eg24s ~/tmp/eg12s ~/tmp/eg6s ~/tmp/eg0s | sort -n -k 4 | cat -n >data
gnuplot<<EOF
set terminal png size 1000,300
set output "compare.png"
set key outside
set style data linespoints
set datafile separator whitespace
set xlabel "datasets (sorted by '24')0
set ylabel "D"
plot \\
  'data' using 1:8 title 'baseline' with lines lc rgb "red", \\
  'data' using 1:6 title '12' with lines, \\
  'data' using 1:5 title '24' with lines lw 3 lc rgb "blue", \\
  'data' using 1:4 title '50' with lines lw 3, \\
  'data' using 1:2 title '200' with lines lw 3 lc rgb "green"
EOF
open compare.png
