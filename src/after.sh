cat $1 |
sort -n |
gawk '
BEGIN { 
   x[3]=20
   x[4]=15
   x[5]=10
   x[6]=5
   x[7]=3
   x[8]=1
 }
 END{ report(b4) }
$1 != b4 {
	report(b4)
             split("", base,"")
	     split("", a,"")
	     b4=$1}
{
  base[NR] = $2
  for(k in x) a[x[k]][NR]= $k
}
function report(x){
  if (NR==1) return
  print("mu",x,mu(base), mu(a[20]), mu(a[15]), mu(a[10]), mu(a[5]),mu(a[3]),mu(a[1]))
  print("sd",x,sd(base), sd(a[20]), sd(a[15]), sd(a[10]), sd(a[5]),sd(a[3]),sd(a[1]))
}

function sd(a) {
  n=int(0.1*asort(a))
  return (a[9*n] - a[n])/2.56 }

function mu(a,    x,sum) {
  for(x in a) sum += a[x]
  return sum/length(a) } ' |
sort -k1,1 -k2,2n 
  
