class people:
    name = ''
    age = 0    
    __weight = 0 		#定义私有属性,私有属性在类外部无法直接访问
   
    def __init__(self,n,a,w): #定义构造函数
        self.name = n			#姓名
        self.age = a			#年龄
        self.__weight = w		#体重定义为私有属性
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
# 实例化类,将自动调用构造函数完成类的初始化
p = people('Sixtang',39,75)
p.speak() 	 #用类的对象p调用speak方法
