import pandas as pd
import numpy as np

activity = pd.read_csv('train_activity.csv').sort_values('day')

# 하루에 두 번 접속한 사람들 값을 합치기
x = activity.values
print(x)

print(activity.columns)

for i in range(0, len(activity) - 1):

    if x[i][0] == x[i + 1][0]:

        if x[i][1] == x[i + 1][1]:

            if x[i][2] == x[i + 1][2]:
                y = []
                p = [x + y for x , y in zip(x[i+1][4:], x[i][4:])] # 행끼리 합
                z = x[i][0:4] + p
                y.append(z)
                x.insert(i, y)
                del x[i + 1], x[i + 2]
            else:
                pass
        else:
            pass
    else:
        pass

a = x
print(a)
print(type(a))
index = activity.columns

df = pd.DataFrame(a)
print(df)

df_new = df.groupby(2).count()
print(df_new.max(0))