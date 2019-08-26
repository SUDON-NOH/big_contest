import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split

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


N_label.loc[N_label["play_char_cnt"] != 0, "NaN_or_Not"] = 1
N_label.loc[N_label["play_char_cnt"] == 0, "NaN_or_Not"] = 0


N_label.loc[N_label["survival_time"] != 64, "sur"] = 0
N_label.loc[N_label["survival_time"] == 64, "sur"] = 1

N_label = N_label.set_index('acc_id')

# =======================================================================================================================

print(N_label.columns)
X_train, X_test, Y_train, Y_test = train_test_split(N_label[['play_char_cnt', 'non_combat_play_time', 'combat_play_time']], N_label['survival_time'],
                                                    test_size = 0.3, random_state= 123)

print(X_train.shape)
print(X_test.shape)

# X_train = np.array(X_train)
# X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)



nb_classes = 64

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.int32, shape = [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print('여기-3')
W = tf.Variable(tf.random_normal([3, nb_classes], name = 'weight'))
b = tf.Variable(tf.random_normal([nb_classes], name = 'bias'))
print('여기-4')
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
print('여기-5')
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                    labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)
print('여기-6')
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
print('여기-7')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('여기-8')
for step in range(60000):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict = {X : X_train, Y : Y_train})
    if step % 5000 == 0:
        print('step:', step,
              'cost_val:', cost_val,
              'W_val', W_val,
              'b_val', b_val)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:X_test, Y:Y_test})

print('Hypothesis:' , h,
      '\nPredict:', p,
      '\nAccuracy:', a)
