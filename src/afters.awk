sort -n ~/tmp/aftersReport.out |
gawk '
BEGIN { 
   x[3]=15
   x[4]=10
   x[5]=5
   x[6]=2
 }
 END{ report(b4) }
$1 != b4 {
	report(b4)
             split("", base,"")
	     split("", a,"")
	     split("", R,"")
	     b4=$1}
{
  base[NR] = $2
  for(k in x) a[x[k]][NR]= $k
  R[NR] = $7
}
function report(x){
  if (NR==1) return
  print("mu",x,mu(base), mu(a[15]), mu(a[10]), mu(a[5]), mu(a[2]),mu(R))
  print("sd",x,sd(base), sd(a[15]), sd(a[10]), sd(a[5]), sd(a[2]),sd(R))
}

function sd(a) {
  n=int(0.1*asort(a))
  return (a[9*n] - a[n])/2.56 }

function mu(a,    x,sum) {
  for(x in a) sum += a[x]
  return sum/length(a) } ' |
sort -k1,1 -k2,2n 
  
