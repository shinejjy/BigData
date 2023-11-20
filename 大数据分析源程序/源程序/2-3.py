'''鸡翁一，值钱五；鸡母一，值钱三；鸡雏三，值钱一；
百钱买百鸡，则翁、母、雏各几何？'''
xj = 1      # xj代表小鸡
while xj <= 100:
    mj = 1   # mj代表母鸡
    while mj <= 100:
        gj = 100-xj-mj
        if xj/3 + mj *3 + gj * 5 == 100 and gj>=0:
            print('小鸡', xj, '母鸡', mj, '公鸡', gj)
        mj += 1
xj += 1
