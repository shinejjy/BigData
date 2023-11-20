from efficient_apriori import apriori 	#导入模块
# 设置事务数据集data
data = [('牛奶','面包','香蕉'),
           ('可乐','面包', '香蕉', '啤酒'),
           ('牛奶','香蕉', '啤酒', '鸡蛋'),
           ('面包', '牛奶', '香蕉', '啤酒'),
           ('面包', '牛奶', '香蕉', '可乐')]
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(data, min_support=0.5,  min_confidence=1)
print(itemsets)   #输出频繁项集  
print(rules)   #输出关联规则
