import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import tensorflow as tf

matplotlib.rcParams['font.family']='Malgun Gothic'   # 한글 사용
matplotlib.rcParams['axes.unicode_minus'] = False

activity = pd.read_csv('train_activity.csv')
label = pd.read_csv('train_label.csv')
# combat = pd.read_csv('train_combat.csv')
# payment = pd.read_csv('train_payment.csv')
# trade = pd.read_csv('train_trade.csv')
pledge = pd.read_csv('train_pledge.csv')

# char_id 기준으로 pledge_id가 바뀌는 사람들 찾기

ple_1 = pledge.groupby(['acc_id', 'char_id', 'pledge_id'], as_index = False).sum()

ple_1.drop(columns = '')