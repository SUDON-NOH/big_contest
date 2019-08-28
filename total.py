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
random.seed(7777)

matplotlib.rcParams['font.family']='Malgun Gothic'   # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

label = pd.read_csv('train_label.csv')


# ======================================================================================================================


activity = pd.read_csv('train_activity.csv')

# activity acc_id로 groupby
# - 평균을 내지 않은 이유 : 평균을 냈을 경우 캐릭터는 많지만
#   한 캐릭터만으로 활동한 사람의 정보가 과소평가 될 가능성이 있음
activity = activity.groupby(['acc_id'], as_index = False).sum()
activity.drop(columns = ['day','char_id'], inplace = True)
# print(activity.head())



# ======================================================================================================================


combat = pd.read_csv('train_combat.csv')

combat.drop(columns = ['day', 'server', 'char_id', 'class'], inplace = True)
combat_a = combat.groupby(['acc_id'], as_index = False).sum()
combat_a.drop(columns = 'level', inplace = True)

combat_b = combat.groupby(['acc_id'], as_index = False).max()
#  combat.groupby('acc_id', as_index = False).sum().sort_values('acc_id')

# acc_id 기준으로 정리

combat_b = combat_b[['acc_id', 'level']]
combat = pd.merge(combat_b, combat_a, how = 'inner', on = 'acc_id')



# ======================================================================================================================


payment = pd.read_csv('train_payment.csv')

payment = payment.groupby('acc_id', as_index = False).sum()
payment.drop(columns = 'day', inplace = True)

payment.rename(columns = {'amount_spent' : 'amount_spent_pay'}, inplace = True)


# ======================================================================================================================


trade = pd.read_csv('train_trade.csv')

# 거래에 참여한 횟수를 기준으로

# 판매자로서 활동한 acc_id
trade_a = trade.groupby('source_acc_id', as_index = False).count()
trade_a = trade_a[['source_acc_id', 'day']]

# 구매자로서 활동한 acc_id
trade_b = trade.groupby('target_acc_id', as_index = False).count()
trade_b = trade_b[['target_acc_id', 'day']]

x = trade_a['day'].sum() - trade_b['day'].sum()
print(x) # 0

trade_a.rename(columns={'source_acc_id':'acc_id',
                        'day':'sell_item_cnt'}, inplace=True)
trade_b.rename(columns={'target_acc_id':'acc_id',
                        'day':'buy_item_cnt'}, inplace=True)

trade = pd.merge(trade_a, trade_b, how = 'outer', on = 'acc_id')

# 실제 데이터 검색
# trade[trade['source_acc_id'] == 6].count()

# 데이터 검증
# trade_a[trade_a['source_acc_id'] == 6]


# ======================================================================================================================


pledge = pd.read_csv('train_pledge.csv')

ple_1 = pledge.groupby(['server', 'pledge_id', 'day'], as_index = False).mean()
ple_1.drop(columns = ['acc_id', 'char_id', 'day'], inplace = True)
ple_1 = ple_1.groupby(['server', 'pledge_id'], as_index = False).sum()

ple_a = pledge.groupby(['acc_id', 'char_id', 'server', 'pledge_id', 'day'], as_index = False).mean()
ple_a = ple_a.groupby(['acc_id', 'char_id', 'server', 'pledge_id'], as_index = False).sum()
ple_a = ple_a[['acc_id', 'char_id', 'server', 'pledge_id']]

pledge = pd.merge(ple_a, ple_1, how = 'outer', on = ['server', 'pledge_id'])
pledge.drop(columns = ['char_id', 'server', 'pledge_id'], inplace = True)
pledge_total = pledge.groupby(['acc_id'], as_index = False).mean()

pledge_total.rename(columns = {'random_attacker_cnt' : 'random_attacker_cnt_plg',
                              'random_defender_cnt' : 'random_defender_cnt_plg',
                              'same_pledge_cnt' : 'same_pledge_cnt_plg',
                              'temp_cnt' : 'temp_cnt_plg',
                              'etc_cnt':'etc_cnt_plg'}, inplace = True)

# ple_1[(ple_1['pledge_id'] == 25467) & (ple_1['server'] == 'aq')]
# ple_a[(ple_a['pledge_id'] == 25467) & (ple_a['server'] == 'aq')]


# ======================================================================================================================


# label + activity
label_a = pd.merge(label, activity, how = 'outer', on = 'acc_id')

# (label + activity) + combat
label_b = pd.merge(label_a, combat, how = 'outer', on = 'acc_id')

# (label + activity + combat) + payment
label_c = pd.merge(label_b, payment, how = 'outer', on = 'acc_id')

# (label + activity + combat + payment) + trade
label_d = pd.merge(label_c, trade, how = 'outer', on = 'acc_id')
label_d = label_d[label_d['survival_time'] >= 1]

# (label + activity + combat + payment + trade) + pledge_total
label_z = pd.merge(label_d, pledge_total, how = 'outer', on = 'acc_id')

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

data = label_z.fillna(0)
print(data.columns)

# ======================================================================================================================

data.loc[data['amount_spent_pay'] == 0, "cash"] = 0
data.loc[data['amount_spent_pay'] > 0, "cash"] = 1

# ======================================================================================================================


# 무과금

data_1 = data[data["cash"] == 0]
print(data_1.count())
data_1_corr = data_1.corr()


# ======================================================================================================================


# 이탈 예측

# sigmoid 활용




# ======================================================================================================================



# 과금
data_2 = data[data["cash"] == 1]
print(data_2.count())
data_2_corr = data_2.corr()