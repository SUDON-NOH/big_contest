import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy as cp

MinMaxScaler = MinMaxScaler()
random.seed(7777)

matplotlib.rcParams['font.family']='Malgun Gothic'   # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

activity = pd.read_csv('train_activity.csv')
label = pd.read_csv('train_label.csv')
# combat = pd.read_csv('train_combat.csv')
# payment = pd.read_csv('train_payment.csv')
# trade = pd.read_csv('train_trade.csv')
pledge = pd.read_csv('train_pledge.csv')

# pledge.sort_values(['pledge_id'])
# 같은 pledge_id, 같은 day 에는 char_id들의 활동 내용이 같다
# 총 pledge ID 의 수
# ple_total = pledge.groupby(['pledge_id'], as_index = False).sum()
# 21860 개

# Pledge ID & Day 로 groupby.mean

ple_1 = pledge.groupby(['pledge_id', 'day'], as_index = False).mean()
ple_1.drop(columns = ['acc_id', 'char_id'], inplace = True)

# 혈맹 활동이 28일동안 꾸준히 기록되지 않음



# pledge ID 로 groupby.sum

ple_2 = ple_1.groupby(['pledge_id'], as_index = False).sum()

# # day의 최대값 406
# day_sum = np.arange(29).sum()
# day_sum 406


# 25, 50, 75 기준으로 나눔
# play_char_cnt
ple_2_pcc = ple_2
print(ple_2_pcc.quantile([.25, .5, .75], axis = 0))
q25, q50, q75 = ple_2_pcc["play_char_cnt"].quantile([.25, .5, .75])
print(q25, q50, q75)


ple_2_pcc.loc[ple_2_pcc["play_char_cnt"] < 0.072179, "play_char_cnt_grade"] = 1
ple_2_pcc.loc[(ple_2_pcc["play_char_cnt"] >= 0.072179) & (ple_2_pcc["play_char_cnt"] < 0.324804), "play_char_cnt_grade"] = 2
ple_2_pcc.loc[(ple_2_pcc["play_char_cnt"] >= 0.324804) & (ple_2_pcc["play_char_cnt"] < 1.660109), "play_char_cnt_grade"] = 3
ple_2_pcc.loc[ple_2_pcc["play_char_cnt"] >= 1.660109, "play_char_cnt_grade"] = 4


# ple_2_pcc 에서 'pledge_id'와 'play_char_cnt_grade'를 추출
print(ple_2_pcc.columns)
new_ple = ple_2_pcc[['pledge_id', 'play_char_cnt_grade']]


ple_a = pledge.groupby(['acc_id', 'char_id', 'pledge_id', 'day'], as_index = False).mean()
ple_a = ple_a.groupby(['acc_id', 'char_id', 'pledge_id'], as_index = False).sum()
ple_a.drop(columns = ['day'], inplace = True)
pled = pd.merge(ple_a, new_ple, how = 'inner', on = 'pledge_id').sort_values('acc_id')
pled = pled.groupby(['acc_id'], as_index = False).mean()
pled.drop(columns = ['char_id', 'pledge_id'], inplace = True)


# 각각의 캐릭터들이 혈맹에서 어떤 활동을 어느 정도 했는지 예상할 수 없기 때문에, 각 캐릭터들이 속한 혈맹의 활동량으로만
# 측정하도록 한다.


# label 과 pledge 를 outer로 결합

label_ple = pd.merge(label, pled, how = 'outer', on = 'acc_id')

# NaN == 0, Not NaN == 1
N_label = label_ple
N_label = N_label.fillna(0)

print(N_label.columns)

N_label[['play_char_cnt']] = MinMaxScaler.fit_transform(N_label[['play_char_cnt']])

# N_label.loc[N_label["play_char_cnt"] != 0, "NaN_or_Not"] = 1
# N_label.loc[N_label["play_char_cnt"] == 0, "NaN_or_Not"] = 0
#
#
# N_label.loc[N_label["survival_time"] != 64, "sur"] = 0
# N_label.loc[N_label["survival_time"] == 64, "sur"] = 1

N_label = N_label.set_index('acc_id')

# =======================================================================================================================

label_activity=pd.merge(label , activity , on='acc_id',how='inner')

ptm = label_activity[['acc_id','day','playtime']]

play_time=ptm.groupby(['acc_id','day'], as_index=False).sum()


# 중복없이 리스트로 뽑기
list(set(play_time.loc[:,'acc_id']))

# acc_id 걸리게 해주기
# play_time[play_time.loc[:,'acc_id'] == '원하는 acc_id']

# 기울기만 뽑기
# cp.polyfit(df['day'], df['playtime'], 1)[0]


# 시작
new_acc=list(set(play_time.loc[:,'acc_id']))
weight = []
for i in new_acc:
    acc = play_time[play_time.loc[:,'acc_id'] == i]
    weight.append(cp.polyfit(acc['day'], acc['playtime'], 1)[0])
print(weight)

# acc_id 와 같은지
# len(W)
# 40000_길이


# acc_id, 와 W 합치기
woals=list(set(play_time.loc[:,'acc_id']))

# 기울기 100으로 키우기
[x*100 for x in weight]

worhkd = [woals,weight]

worhkd1 = pd.DataFrame(worhkd).T
print(worhkd1)

# rename
worhkd1.rename(columns={0:'acc_id'})
ehsl = worhkd1.rename(columns={1:'weight',0:'acc_id'})
print(ehsl)

# float를 int형으로 바꾸기
ehsl['acc_id'] = ehsl['acc_id'].astype(int)


# 데이터 합치기
result = pd.merge(ehsl,label,how='inner')
print(result.columns)

# =======================================================================================================================

result.drop(columns = ['survival_time' , 'amount_spent'], inplace = True)

N_label = pd.merge(N_label, result, how = 'inner', on = 'acc_id')

print(N_label)

# =======================================================================================================================


X_train, X_test, Y_train, Y_test = train_test_split(N_label[['play_char_cnt', 'weight']], N_label['survival_time'],
                                                    test_size = 0.3, random_state= 123)


X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train.reshape(-1, 2)
X_test = X_test.reshape(-1, 2)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

nb_classes = 64

X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.int32, shape = [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([2, nb_classes], name = 'weight'))
b = tf.Variable(tf.random_normal([nb_classes], name = 'bias'))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                    labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(50000):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict = {X : X_train, Y : Y_train})
    if step % 5000 == 0:
        print('step:', step,
              'cost_val:', cost_val)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:X_test, Y:Y_test})

print('Hypothesis:' , h,
      '\nPredict:', p,
      '\nAccuracy:', a)
