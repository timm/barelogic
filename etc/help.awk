BEGIN { 
  COLOR= "\033[36m" #31=red,32=green,33=brown,34=blue,35=purple,36=cyan.37=white
  RESET= "\033[0m"     
  FS   = ":.*?## "        
  print "\nmake [WHAT]" 
}
/^[^ \t].*##/ {          
  printf("   %s%-15s%s : %s\n", COLOR, $1, RESET, $2) | "sort"  
}
