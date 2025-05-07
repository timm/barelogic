FNR==1{ x=count_alnum_ends($0); y= NF-x}
END { printf("%6s %6s %6s %s\n", x,y,NR,FILENAME)}

function count_alnum_ends(str,    n, i, a, count) {
  n = split(str, a, ",")
  count = 0
  for (i = 1; i <= n; i++)
    if (a[i] ~ /[a-zA-Z0-9]$/)
      count++
  return count
}
