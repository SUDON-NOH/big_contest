import pandas as pd


activity = pd.read_csv('train_activity.csv')

label = pd.read_csv('train_label.csv')

print(activity)
print(label)

print(label.values)
label_1 = label[label['survival_time'] < 8]   # 1일 ~ 7일 그룹
label_2 = label[(label['survival_time'] >= 8)&(label['survival_time'] < 15)]    # 8일 ~ 14일 그룹
label_3 = label[(label['survival_time'] >= 15)&(label['survival_time'] < 22)]
label_4 = label[(label['survival_time'] >= 22)&(label['survival_time'] < 29)]
label_5 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 36)]
label_6 = label[(label['survival_time'] >= 36)&(label['survival_time'] < 43)]
label_7 = label[(label['survival_time'] >= 29)&(label['survival_time'] < 50)]
label_8 = label[(label['survival_time'] >= 50)&(label['survival_time'] < 57)]
label_9 = label[(label['survival_time'] >= 57)&(label['survival_time'] < 64)]
label_10 = label[(label['survival_time'] == 64)]

# label_1_id = label['acc_id']
# llabel_1_id = label_1_id.tolist()   # acc_id 를 리스트로 만듦
# print(llabel_1_id)
# print(type(llabel_1_id))

# print(activity.columns)



x1 = pd.merge(label_1, activity, how = 'inner', on = 'acc_id')  # 1일 ~ 7일 그룹 사람들을 구별해서 activity와 merge
x2 = pd.merge(label_2, activity, how = 'inner', on = 'acc_id')
x3 = pd.merge(label_3, activity, how = 'inner', on = 'acc_id')
x4 = pd.merge(label_4, activity, how = 'inner', on = 'acc_id')
x5 = pd.merge(label_5, activity, how = 'inner', on = 'acc_id')
x6 = pd.merge(label_6, activity, how = 'inner', on = 'acc_id')
x7 = pd.merge(label_7, activity, how = 'inner', on = 'acc_id')
x8 = pd.merge(label_8, activity, how = 'inner', on = 'acc_id')
x9 = pd.merge(label_9, activity, how = 'inner', on = 'acc_id')
x10 = pd.merge(label_10, activity, how = 'inner', on = 'acc_id')



a1 = x1.drop(columns = 'amount_spent') # amount_spent 삭제
a2 = x2.drop(columns = 'amount_spent')
a3 = x3.drop(columns = 'amount_spent')
a4 = x4.drop(columns = 'amount_spent')
a5 = x5.drop(columns = 'amount_spent')
a6 = x6.drop(columns = 'amount_spent')
a7 = x7.drop(columns = 'amount_spent')
a8 = x8.drop(columns = 'amount_spent')
a9 = x9.drop(columns = 'amount_spent')
a10 = x10.drop(columns = 'amount_spent')



y1 = a1.groupby('acc_id').mean()[['fishing']]
y2 = a2.groupby('acc_id').mean()[['fishing']]
y3 = a3.groupby('acc_id').mean()[['fishing']]
y4 = a4.groupby('acc_id').mean()[['fishing']]
y5 = a5.groupby('acc_id').mean()[['fishing']]
y6 = a6.groupby('acc_id').mean()[['fishing']]
y7 = a7.groupby('acc_id').mean()[['fishing']]
y8 = a8.groupby('acc_id').mean()[['fishing']]
y9 = a9.groupby('acc_id').mean()[['fishing']]
y10 = a10.groupby('acc_id').mean()[['fishing']]

xx = y1.sort_values(["acc_id"], ascending=[False])

z1 = y1.sort_values(["fishing"], ascending=[False])
z2 = y2.sort_values(["fishing"], ascending=[False])
z3 = y3.sort_values(["fishing"], ascending=[False])
z4 = y4.sort_values(["fishing"], ascending=[False])
z5 = y5.sort_values(["fishing"], ascending=[False])
z6 = y6.sort_values(["fishing"], ascending=[False])
z7 = y7.sort_values(["fishing"], ascending=[False])
z8 = y8.sort_values(["fishing"], ascending=[False])
z9 = y9.sort_values(["fishing"], ascending=[False])
z10 = y10.sort_values(["fishing"], ascending=[False])

y


