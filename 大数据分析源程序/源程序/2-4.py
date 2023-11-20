import time
count = 0
while count < 3:		#程序将每隔1秒输出还剩几秒
   print ("还剩 %d 秒"%(3-count)) 		#注意用%连接格式字符串和值
   count = count + 1
   time.sleep(1)   #延时1秒
else:
   print (" 发射！")
