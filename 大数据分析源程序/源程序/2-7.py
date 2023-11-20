def Title(string,n) :
   if len(string)>n:
       return string[0:n]+"……"
   else:
       return string
a=Title('航空母舰已经下水入列！',8)
print(a)
