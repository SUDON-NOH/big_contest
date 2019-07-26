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
label_7 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 50)]
label_8 = label[(label['survival_time'] >= 50)&(label['survival_time'] < 57)]
label_9 = label[(label['survival_time'] >= 57)&(label['survival_time'] < 64)]
label_10 = label[(label['survival_time'] == 64)]



fishing_group = activity.groupby('acc_id').sum()[['fishing']]

x1 = pd.merge(label_1, fishing_group, how = 'inner', on = 'acc_id')
x2 = pd.merge(label_2, fishing_group, how = 'inner', on = 'acc_id')
x3 = pd.merge(label_3, fishing_group, how = 'inner', on = 'acc_id')
x4 = pd.merge(label_4, fishing_group, how = 'inner', on = 'acc_id')
x5 = pd.merge(label_5, fishing_group, how = 'inner', on = 'acc_id')
x6 = pd.merge(label_6, fishing_group, how = 'inner', on = 'acc_id')
x7 = pd.merge(label_7, fishing_group, how = 'inner', on = 'acc_id')
x8 = pd.merge(label_8, fishing_group, how = 'inner', on = 'acc_id')
x9 = pd.merge(label_9, fishing_group, how = 'inner', on = 'acc_id')
x10 = pd.merge(label_10, fishing_group, how = 'inner', on = 'acc_id')

print('1주일: ',x1['fishing'].mean())
print('2주일: ',x2['fishing'].mean())
print('3주일: ',x3['fishing'].mean())
print('4주일: ',x4['fishing'].mean())
print('5주일: ',x5['fishing'].mean())
print('6주일: ',x6['fishing'].mean())
print('7주일: ',x7['fishing'].mean())
print('8주일: ',x8['fishing'].mean())
print('9주일: ',x9['fishing'].mean())
print('10주일: ',x10['fishing'].mean())

