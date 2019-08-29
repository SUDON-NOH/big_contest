# ======================================================================================================================
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

# ======================================================================================================================

scaler = MinMaxScaler()
random.seed(54654)

matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

label = pd.read_csv('train_label.csv')

# ======================================================================================================================
print("process - 1")

activity = pd.read_csv('train_activity.csv')

# activity acc_id로 groupby
# - 평균을 내지 않은 이유 : 평균을 냈을 경우 캐릭터는 많지만
#   한 캐릭터만으로 활동한 사람의 정보가 과소평가 될 가능성이 있음
activity = activity.groupby(['acc_id'], as_index=False).sum()
activity.drop(columns=['day', 'char_id'], inplace=True)
# print(activity.head())


# ======================================================================================================================


combat = pd.read_csv('train_combat.csv')

combat.drop(columns=['day', 'server', 'char_id', 'class'], inplace=True)
combat_a = combat.groupby(['acc_id'], as_index=False).sum()
combat_a.drop(columns='level', inplace=True)

combat_b = combat.groupby(['acc_id'], as_index=False).max()
#  combat.groupby('acc_id', as_index = False).sum().sort_values('acc_id')

# acc_id 기준으로 정리

combat_b = combat_b[['acc_id', 'level']]
combat = pd.merge(combat_b, combat_a, how='inner', on='acc_id')

# ======================================================================================================================


payment = pd.read_csv('train_payment.csv')

payment = payment.groupby('acc_id', as_index=False).sum()
payment.drop(columns='day', inplace=True)

payment.rename(columns={'amount_spent': 'amount_spent_pay'}, inplace=True)

# ======================================================================================================================


trade = pd.read_csv('train_trade.csv')

# 거래에 참여한 횟수를 기준으로

# 판매자로서 활동한 acc_id
trade_a = trade.groupby('source_acc_id', as_index=False).count()
trade_a = trade_a[['source_acc_id', 'day']]

# 구매자로서 활동한 acc_id
trade_b = trade.groupby('target_acc_id', as_index=False).count()
trade_b = trade_b[['target_acc_id', 'day']]

x = trade_a['day'].sum() - trade_b['day'].sum()
# print(x) # 0

trade_a.rename(columns={'source_acc_id': 'acc_id',
                        'day': 'sell_item_cnt'}, inplace=True)
trade_b.rename(columns={'target_acc_id': 'acc_id',
                        'day': 'buy_item_cnt'}, inplace=True)

trade = pd.merge(trade_a, trade_b, how='outer', on='acc_id')

# 실제 데이터 검색
# trade[trade['source_acc_id'] == 6].count()

# 데이터 검증
# trade_a[trade_a['source_acc_id'] == 6]


# ======================================================================================================================


pledge = pd.read_csv('train_pledge.csv')

ple_1 = pledge.groupby(['server', 'pledge_id', 'day'], as_index=False).mean()
ple_1.drop(columns=['acc_id', 'char_id', 'day'], inplace=True)
ple_1 = ple_1.groupby(['server', 'pledge_id'], as_index=False).sum()

ple_a = pledge.groupby(['acc_id', 'char_id', 'server', 'pledge_id', 'day'], as_index=False).mean()
ple_a = ple_a.groupby(['acc_id', 'char_id', 'server', 'pledge_id'], as_index=False).sum()
ple_a = ple_a[['acc_id', 'char_id', 'server', 'pledge_id']]

pledge = pd.merge(ple_a, ple_1, how='outer', on=['server', 'pledge_id'])
pledge.drop(columns=['char_id', 'server', 'pledge_id'], inplace=True)
pledge_total = pledge.groupby(['acc_id'], as_index=False).mean()

pledge_total.rename(columns={'random_attacker_cnt': 'random_attacker_cnt_plg',
                             'random_defender_cnt': 'random_defender_cnt_plg',
                             'same_pledge_cnt': 'same_pledge_cnt_plg',
                             'temp_cnt': 'temp_cnt_plg',
                             'etc_cnt': 'etc_cnt_plg'}, inplace=True)

# ple_1[(ple_1['pledge_id'] == 25467) & (ple_1['server'] == 'aq')]
# ple_a[(ple_a['pledge_id'] == 25467) & (ple_a['server'] == 'aq')]


# ======================================================================================================================
print("process - 2")

# label + activity
label_a = pd.merge(label, activity, how='outer', on='acc_id')

# (label + activity) + combat
label_b = pd.merge(label_a, combat, how='outer', on='acc_id')

# (label + activity + combat) + payment
label_c = pd.merge(label_b, payment, how='outer', on='acc_id')

# (label + activity + combat + payment) + trade
label_d = pd.merge(label_c, trade, how='outer', on='acc_id')
label_d = label_d[label_d['survival_time'] >= 1]

# (label + activity + combat + payment + trade) + pledge_total
label_z = pd.merge(label_d, pledge_total, how='outer', on='acc_id')

data = label_z.fillna(0)
print(data.columns)

data[['playtime', 'npc_kill',
      'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
      'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
      'enchant_count', 'level', 'pledge_cnt', 'random_attacker_cnt',
      'random_defender_cnt', 'temp_cnt', 'same_pledge_cnt', 'etc_cnt',
      'num_opponent', 'amount_spent_pay', 'sell_item_cnt', 'buy_item_cnt',
      'play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt',
      'random_attacker_cnt_plg', 'random_defender_cnt_plg',
      'same_pledge_cnt_plg', 'temp_cnt_plg', 'etc_cnt_plg',
      'combat_play_time', 'non_combat_play_time']] = \
    scaler.fit_transform(data[['playtime', 'npc_kill',
                               'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
                               'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
                               'enchant_count', 'level', 'pledge_cnt', 'random_attacker_cnt',
                               'random_defender_cnt', 'temp_cnt', 'same_pledge_cnt', 'etc_cnt',
                               'num_opponent', 'amount_spent_pay', 'sell_item_cnt', 'buy_item_cnt',
                               'play_char_cnt', 'combat_char_cnt', 'pledge_combat_cnt',
                               'random_attacker_cnt_plg', 'random_defender_cnt_plg',
                               'same_pledge_cnt_plg', 'temp_cnt_plg', 'etc_cnt_plg',
                               'combat_play_time', 'non_combat_play_time']])

data = data.fillna(0)
print(data.columns)

# ======================================================================================================================

data.loc[data['amount_spent_pay'] == 0, "cash"] = 0
data.loc[data['amount_spent_pay'] > 0, "cash"] = 1

# ======================================================================================================================


# 무과금

data_1 = data[data["cash"] == 0]
print(data_1.count())
data_1_corr = data_1.corr()

data_1.loc[data_1['survival_time'] < 64, "survived"] = 0
data_1.loc[data_1['survival_time'] == 64, "survived"] = 1

data_1_2 = data_1.corr()

# ======================================================================================================================

# 무과금 + 이탈
print("process - 3")
data_2 = data_1[data_1["survived"] == 0]

data_2_1 = data_2.corr()
"""

# 서로 반비례관계에 있을 시 제외

survival_time :
    playtime # 0.19
        npc_kill                # 0.5
        party_exp               # 0.17
        death                   # 0.2
        revive                  # 0.15
        private_shop            # 0.33
        level                   # 0.13
        random_defender_cnt     # 0.37
        etc_cnt                 # 0.15
        num_opponent            # 0.17
        sell_item_cnt           # 0.15
        random_defender_cnt_plg # 0.15
        combat_play_time        # 0.29
        non_combat_play_time    # 0.33
    npc_kill # 0.15
        rich_monster            # 0.12
        death                   # 0.11
        level                   # 0.40
        random_defender_cnt     # 0.26
        etc_cnt                 # 0.24
        num_opponent            # 0.23
    rich_monster
        solo_exp                # 0.47
        quest_exp               # 0.30
        death                   # 0.22
        revive                  # 0.23
        level                   # 0.17
        num_opponent            # 0.24
        play_char_cnt           # 0.14
        pledge_combat_cnt       # 0.14
        random_defender_cnt_plg # 0.18
    level
        combat_play_time        # 0.31
    num_opponent
        pledge_cnt              # 0.41

"""
# print(data_2.columns)
x_e = data_1[['acc_id', 'playtime', 'npc_kill',
              'solo_exp', 'quest_exp',
              'private_shop', 'level', 'pledge_cnt', 'random_defender_cnt', 'etc_cnt',
              'num_opponent', 'sell_item_cnt',
              'combat_play_time', 'non_combat_play_time']]
y_e = data_1[['acc_id', 'survival_time']]

x = x_e.set_index('acc_id')
y = y_e.set_index('acc_id')

print(y.shape)
print(x.shape)
print(y.columns)
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=8431)

print(type(X_train))
print(type(Y_train))
print(type(X_test))
print(type(Y_test))





# Build Graph
nb_classes = 64
X = tf.placeholder(tf.float32, shape=[None, 13])
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Y 값을 one_hot encoding으로 변환, Y값은 반드시 int형으로 입력
Y_one_hot = tf.one_hot(Y, nb_classes)                # [None, 1, 7]
print(Y_one_hot)                                     # shape=(?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # numpy에서 행을 자동으로 크기에 맞게 다시 조절한다 "-1"
print(Y_one_hot)                                     # [None, 1]


W = tf.Variable(tf.random_normal([13, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')



# hypothesis
logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

# cost function
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                 labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)


# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습단계(start learning)

for step in range(200001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict = {X : X_train, Y : Y_train})
    if step % 10000 == 0:
        print('\nstep:', step,
              '\ncost_val:', cost_val,)


# 검증단계(test, 정확도 측정)
# Accuracy Computation(정확도 계산)
predict = tf.argmax(hypothesis, 1) # 1:행단위 # 예측값 행에서 가장 큰 값, 확률값을 구한다.
# predict = tf.argmax(hypothesis,) # 0:열단위

correct_predict = tf.equal(predict, tf.argmax(Y_one_hot, 1))  # predict와 y 값을 비교한다.
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32)) # 숫자로 바꾼 뒤 평균을 낸다.

h, p, a = sess.run([hypothesis, predict, accuracy],feed_dict={X:X_test, Y:Y_test})

print('Hypothesis:' , h,
      '\nPredict:',p,
      '\nAccuracy:', a)

