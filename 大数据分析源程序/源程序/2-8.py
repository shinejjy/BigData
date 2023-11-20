class Jzt:          #定义Jzt类
    shape="*"       #成员变量（属性）
    row=5
    def draw(self,row,shape):       #成员函数（方法）
        for i in range(row):
            for j in range(-1,i):
                print(shape,end=' ')
            print()   
#声明对象p
p=Jzt()  
shape="%"
row=4
p.draw(row,shape)   #调用类的方法
print(p.shape,p.row)  #调用类的属性
