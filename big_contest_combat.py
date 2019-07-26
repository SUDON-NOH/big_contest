import pandas as pd


combat = pd.read_csv('train_combat.csv')
label = pd.read_csv('train_label.csv')

# combat 파일 acc_id 의 최소값을 값는 id 확인
print(combat['acc_id'].min())
print(label.values)





# 기간별로 나눈 유저 아이디 ( 'acc_id' )
# 강사님한테 여쭤보기 - 함수로 표현, 데이터 프레임 모양을 훼손시키지 않고 하는법 // 이중 list or dictionary

label_1 = label[label['survival_time'] < 8]
label_2 = label[(label['survival_time'] >= 8)&(label['survival_time'] < 15)]
label_3 = label[(label['survival_time'] >= 15)&(label['survival_time'] < 22)]
label_4 = label[(label['survival_time'] >= 22)&(label['survival_time'] < 29)]
label_5 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 36)]
label_6 = label[(label['survival_time'] >= 36)&(label['survival_time'] < 43)]
label_7 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 50)]
label_8 = label[(label['survival_time'] >= 50)&(label['survival_time'] < 57)]
label_9 = label[(label['survival_time'] >= 57)&(label['survival_time'] < 64)]
label_10 = label[(label['survival_time'] == 64)]






# 유저 아이디별 & 캐릭터별 combat 관련 groupby

combat_group = combat.groupby('acc_id').sum()[[
    'pledge_cnt','random_attacker_cnt','random_defender_cnt',
    'temp_cnt','same_pledge_cnt', 'etc_cnt', 'num_opponent']]

combat_group2 = combat.groupby('char_id').sum()[[
    'pledge_cnt','random_attacker_cnt','random_defender_cnt',
    'temp_cnt','same_pledge_cnt', 'etc_cnt', 'num_opponent']]


xx = combat_group.sort_values(['acc_id'], ascending=[False])
xx2 = combat_group2.sort_values(['char_id'], ascending=[False])

# 캐릭터보다 유저로 봤을 때 더 분산도가 큼
print(xx.var())
print(xx2.var())


                                        # 혈맹에 가입하지 않는 ID 들은 train_pledge에 나오지 않는다.

# 혈맹간 전투 참여한 유저는 이탈가능성이 현저히 낮을까?

combat_group3 = combat.groupby('acc_id').sum()[['pledge_cnt']]



x1 = pd.merge(label_1, combat_group3, how = 'inner', on = 'acc_id')
x2 = pd.merge(label_2, combat_group3, how = 'inner', on = 'acc_id')
x3 = pd.merge(label_3, combat_group3, how = 'inner', on = 'acc_id')
x4 = pd.merge(label_4, combat_group3, how = 'inner', on = 'acc_id')
x5 = pd.merge(label_5, combat_group3, how = 'inner', on = 'acc_id')
x6 = pd.merge(label_6, combat_group3, how = 'inner', on = 'acc_id')
x7 = pd.merge(label_7, combat_group3, how = 'inner', on = 'acc_id')
x8 = pd.merge(label_8, combat_group3, how = 'inner', on = 'acc_id')
x9 = pd.merge(label_9, combat_group3, how = 'inner', on = 'acc_id')
x10 = pd.merge(label_10, combat_group3, how = 'inner', on = 'acc_id')




print('1주일: ',x1['pledge_cnt'].mean())
print('2주일: ',x2['pledge_cnt'].mean())
print('3주일: ',x3['pledge_cnt'].mean())
print('4주일: ',x4['pledge_cnt'].mean())
print('5주일: ',x5['pledge_cnt'].mean())
print('6주일: ',x6['pledge_cnt'].mean())
print('7주일: ',x7['pledge_cnt'].mean())
print('8주일: ',x8['pledge_cnt'].mean())
print('9주일: ',x9['pledge_cnt'].mean())
print('10주일: ',x10['pledge_cnt'].mean())
