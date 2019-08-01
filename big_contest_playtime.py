import pandas as pd



activity = pd.read_csv('train_activity.csv')
label = pd.read_csv('train_label.csv')

# combat 파일 acc_id 의 최소값을 값는 id 확인
print(activity['acc_id'].min())
print(label.values)

label_1 = label[label['survival_time'] < 8]
label_2 = label[(label['survival_time'] >= 8)&(label['survival_time'] < 15)]
label_3 = label[(label['survival_time'] >= 15)&(label['survival_time'] < 22)]
label_4 = label[(label['survival_time'] >= 22)&(label['survival_time'] < 29)]
label_5 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 36)]
label_6 = label[(label['survival_time'] >= 36)&(label['survival_time'] < 43)]
label_7 = label[(label['survival_time'] >= 43)&(label['survival_time'] < 50)]
label_8 = label[(label['survival_time'] >= 50)&(label['survival_time'] < 57)]
label_9 = label[(label['survival_time'] >= 57)&(label['survival_time'] < 64)]
label_10 = label[(label['survival_time'] == 64)]

playtime_group = activity.groupby('acc_id').sum()[['playtime']]

x1 = pd.merge(label_1, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x2 = pd.merge(label_2, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x3 = pd.merge(label_3, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x4 = pd.merge(label_4, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x5 = pd.merge(label_5, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x6 = pd.merge(label_6, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x7 = pd.merge(label_7, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x8 = pd.merge(label_8, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x9 = pd.merge(label_9, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])
x10 = pd.merge(label_10, playtime_group, how = 'inner', on = 'acc_id').sort_values(['playtime'], ascending = [False])


print('playtime mean 1주일: ',x1['playtime'].mean())
print('playtime mean 2주일: ',x2['playtime'].mean())
print('playtime mean 3주일: ',x3['playtime'].mean())
print('playtime mean 4주일: ',x4['playtime'].mean())
print('playtime mean 5주일: ',x5['playtime'].mean())
print('playtime mean 6주일: ',x6['playtime'].mean())
print('playtime mean 7주일: ',x7['playtime'].mean())
print('playtime mean 8주일: ',x8['playtime'].mean())
print('playtime mean 9주일: ',x9['playtime'].mean())
print('playtime mean 10주일: ',x10['playtime'].mean())

print('playtime sum 1주일: ',x1['playtime'].sum())
print('playtime sum 2주일: ',x2['playtime'].sum())
print('playtime sum 3주일: ',x3['playtime'].sum())
print('playtime sum 4주일: ',x4['playtime'].sum())
print('playtime sum 5주일: ',x5['playtime'].sum())
print('playtime sum 6주일: ',x6['playtime'].sum())
print('playtime sum 7주일: ',x7['playtime'].sum())
print('playtime sum 8주일: ',x8['playtime'].sum())
print('playtime sum 9주일: ',x9['playtime'].sum())
print('playtime sum 10주일: ',x10['playtime'].sum())



print('1주일: ',x1['acc_id'].count())
print('2주일: ',x2['acc_id'].count())
print('3주일: ',x3['acc_id'].count())
print('4주일: ',x4['acc_id'].count())
print('5주일: ',x5['acc_id'].count())
print('6주일: ',x6['acc_id'].count())
print('7주일: ',x7['acc_id'].count())
print('8주일: ',x8['acc_id'].count())
print('9주일: ',x9['acc_id'].count())
print('10주일: ',x10['acc_id'].count())