import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 필요한 데이터 불러오기
activity = pd.read_csv('train_activity.csv')
combat = pd.read_csv('train_combat.csv')
label = pd.read_csv('train_label.csv')

activity = activity.sort_values('char_id')



# 캐릭터 아이디별 생존기간
char_life = pd.merge(activity, label, how = 'inner', on = 'acc_id')



# char_id와 survival time만 남기고 모든 컬럼 삭제
char_life = char_life[['char_id', 'survival_time']]
char_life = char_life.groupby('char_id').max()['survival_time']



# activity char group_by
char_act = activity.groupby('char_id').sum()
char_act.drop(columns = ['day', 'acc_id'], inplace = True)


# char_id를 기준으로 activity에서 groupby 된 데이터와 병합

char_act = pd.merge(char_act, char_life, how = 'inner', on = 'char_id')


# 일별 플레이 수
act_day_c = activity.groupby('char_id').count()

# 28일 이상인 데이터
x = act_day_c[act_day_c['day'] > 28]
print(x)


act_day_c = act_day_c['day']


# activity에 플레이 일 수(day)를 merge
act_day = pd.merge(char_act, act_day_c, how = 'inner', on = 'char_id')

# 플레이 일 수(day)의 이름을 변경
act_day = act_day.rename(columns = {'day':'day_cnt'})

# char_id 기준 level 추출
combat_lv = combat.groupby('char_id').max()['level']

# act_day 에 level을 추가
act_day = pd.merge(act_day, combat_lv, how = 'inner', on = 'char_id')

