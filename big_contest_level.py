import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'   # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

activity = pd.read_csv('train_activity.csv')
label = pd.read_csv('train_label.csv')
combat = pd.read_csv('train_combat.csv')

# Level 별 이탈 현황 그래프 만들기
label.loc[label["survival_time"] == 64 , "survived"] = 1
label.loc[label["survival_time"] < 64 , "survived"] = 0

# activity와 label을 우선적으로 merge
meg_act_lab = pd.merge(label, activity, how = 'inner', on = 'acc_id')
print(meg_act_lab.head())
print(meg_act_lab.columns)
print(label.columns)
print(combat.columns)

# char_id 기준으로 groupby.sum() / activity에서 사용할 columns 추출
activity_group = activity.groupby('char_id').sum()[['playtime', 'npc_kill', 'solo_exp', 'party_exp',
                                                  'quest_exp','boss_monster', 'death', 'revive', 'exp_recovery',
                                                  'fishing','private_shop', 'game_money_change', 'enchant_count']]
print(activity_group.count())

# char_id 기준으로 groupby.sum() / combat에서 사용할 columns 추출
combat_group = combat.groupby('char_id').sum()[['pledge_cnt','random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
                                                  'same_pledge_cnt','etc_cnt','num_opponent']]

# char_id 기준 level 추출
combat_lv = combat.groupby('char_id').max()['level']
print(combat_lv.count())

# char_id 를 label 에 추가하는 과정
selected_group = meg_act_lab[['acc_id', 'char_id', 'survival_time', 'amount_spent', 'survived']]

# char_id를 기준으로 level 과 acc_id 를 추가하는 과정
# ( activity_group에 직접 추가를 해도 되지만, acc_id와 survival_time 이 필요하기 때문에 과정을 한번 더 거친다.)
selected_group2 = pd.merge(selected_group, combat_lv, how = 'inner', on = 'char_id')

# 이 부분에서 중복되는 것을 막기 위해 char_id를 기준으로 최대값들을 구해준다.
selected = selected_group2.groupby('char_id').max()

# 앞서 activity에서 사용할 colums을 추출한 데이터와 char_id를 기준으로 level, survival_time 등을 가져온 데이터를 merge
activity_group3 = pd.merge(activity_group, selected, how = 'inner', on = 'char_id')

combat_group1 = pd.merge(combat_group, selected, how = 'inner', on = 'char_id')


"""
범주     레벨
 0  :  1 ~ 4
 1  :  5 ~ 9
 2  : 10 ~ 14
 3  : 15 ~ 19
 4  : 20 ~ 24
 5  : 25 ~ 29
 6  : 30 ~ 34
 7  : 35 ~ 39
 8  : 40 ~ 44
 9  : 45 ~ 49
 10 : 50 ~ 54
 11 : 55 ~ 59
 12 : 60 ~ 64
 13 : 65 ~ 69
 14 : 70 ~ 74
 15 : 75 ~ 79
 16 : 80 ~ 84
 17 : 85 이상
"""

corr_level_active = activity_group3.corr()
corr_level_combat = combat_group1.corr()

# 그래프
sns.countplot(data = activity_group3, x="level", hue="survived")
plt.grid(True)
plt.xlabel('level 범주')
plt.ylabel('Survival count')
plt.title('생존/비생존 숫자')
plt.show()

print(activity_group3.count())
