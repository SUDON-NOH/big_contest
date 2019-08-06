import lightgbm as lgb
import pandas as pd
import numpy as np

np.random.seed(0)

def get_int_rand(min, max):
    value = np.random.randint(min, high = max + 1, size = 1)
    return int(value[0])

def get_rand(min, max):
    value = np.random.rand(1)*(max-min)+min
    return float(value[0])

# make train set
df = pd.DataFrame([],columns = ['x1', 'x2', 'y1','y2'])

for i in range(1000):
    x1 = get_int_rand(0, 1)
    x2 = get_int_rand(0, 1)
    if x1 == x2:
        y1 = 0
        y2 = 1
    else:
        y1 = 1
        y2 = 0
    x1 = get_rand(x1 - 0.3, x1 + 0.3)
    x2 = get_rand(x2 - 0.3, x2 + 0.3)

    df = df.append(pd.DataFrame(np.array([x1, x2,
                                          int(y1), int(y2)]).reshape(1,4), columns = ['x1', 'x2','y1','y2']))

df.reset_index(inplace = True, drop = True)
print(df.head())

df.to_csv('train.csv', encoding = 'utf-8', index = False)