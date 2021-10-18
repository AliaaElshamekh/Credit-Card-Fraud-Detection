import pandas as pd
import json
import  datetime
import  re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import altair as alt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import scipy.stats as sc
import sklearn
plt.style.use('ggplot')
import seaborn as sns
import warnings
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

import lightgbm as lgb
#from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import multiprocessing
import gc

print("Hello")
warnings.simplefilter("ignore")

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

train_dataset_url = 'fraud.csv'
train_data = pd.read_csv(train_dataset_url)

test_dataset_url ='fraud_test_sample.csv'
test_data = pd.read_csv(test_dataset_url)
contingency_table=pd.crosstab(train_data["isFraud"],train_data)
print('contingency_table :-\n',contingency_table)

plt.figure(1)
ax = sns.countplot(x="DeviceType", data=train_data)
ax.set_title('DeviceType', fontsize=14)
plt.show()

print (train_data.shape)
print (test_data.shape)

print('Train dataset has '+ str({train_data.shape[0]}) +'rows and '+str({train_data.shape[1]})+' columns.')
print('Test dataset has'+ str({test_data.shape[0]}) +'rows and' + str({test_data.shape[1]}) +'columns.')

print (train_data.head())


train_data.describe(include='all')

many_null_cols = [col for col in train_data.columns if train_data[col].isnull().sum() / train_data.shape[0] > 0.9]
many_null_cols_test = [col for col in test_data.columns if test_data[col].isnull().sum() / test_data.shape[0] > 0.9]

count_classes = pd.value_counts(train_data['isFraud'], sort = True).sort_index()
count_classes.plot(kind = 'bar',color=color_pal[1])
plt.figure(1)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

print(train_data.dtypes)
#
x_pro = (len(train_data)-train_data.count())/len(train_data)*100
x_pro_test = (len(test_data)-test_data.count())/len(test_data)*100

one_value_cols = [col for col in train_data.columns if train_data[col].nunique() <= 1]
one_value_cols_test = [col for col in test_data.columns if test_data[col].nunique() <= 1]
one_value_cols == one_value_cols_test


print('There are'+  str({train_data.isnull().any().sum()}) +'columns in train dataset with missing values.')

one_value_cols = [col for col in train_data.columns if train_data[col].nunique() <= 1]
one_value_cols_test = [col for col in test_data.columns if test_data[col].nunique() <= 1]
one_value_cols == one_value_cols_test

print('There are '+ str({len(one_value_cols)})+' columns in train dataset with one unique value.')
print('There are'+  str({len(one_value_cols_test)})+' columns in test dataset with one unique value.')




many_null_cols = [col for col in train_data.columns if train_data[col].isnull().sum() / train_data.shape[0] > 0.9]
many_null_cols_test = [col for col in test_data.columns if test_data[col].isnull().sum() / test_data.shape[0] > 0.9]

big_top_value_cols = [col for col in train_data.columns if train_data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test_data.columns if test_data[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]


cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))
cols_to_drop.remove('isFraud')
len(cols_to_drop)

train= train_data.drop(cols_to_drop, axis=1)
test = test_data.drop(cols_to_drop, axis=1)

def percent_na(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing},index=None)
    #missing_value_df.sort_values('percent_missing', inplace=True)
    missing_value_df=missing_value_df.reset_index().drop('index',axis=1)
    return missing_value_df
train_transaction_na = percent_na(train_data)

pd.options.display.max_colwidth =300
col_na_group= train_transaction_na.groupby('percent_missing')['column_name'].unique().reset_index()
num_columns=[]
for i in range(len(col_na_group)):
    num_columns.append(len(col_na_group.column_name[i]))
col_na_group['num_columns']=num_columns
col_na_group = col_na_group.loc[(col_na_group['num_columns']>1) & (col_na_group['percent_missing']>0),].sort_values(by='percent_missing',ascending=False).reset_index()
col_na_group



# data.ProductCD  .unique()
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# sel.fit_transform(data)

print(x_pro)

plt.figure(2)
train_data['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test_data['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()

plt.figure(3)
ax = train_data.plot(x='TransactionDT',
                       y='TransactionAmt',
                       kind='scatter',
                       alpha=0.01,
                       label='TransactionAmt-train',
                       title='Train and test Transaction Amounts by Time (TransactionDT)',
                       ylim=(0, 5000),
                       figsize=(15, 5))
test_data.plot(x='TransactionDT',
                      y='TransactionAmt',
                      kind='scatter',
                      label='TransactionAmt-test',
                      alpha=0.01,
                      color=color_pal[1],
                       ylim=(0, 5000),
                      ax=ax)
# Plot Fraud as Orange
train_data.loc[train_data['isFraud'] == 1] \
    .plot(x='TransactionDT',
         y='TransactionAmt',
         kind='scatter',
         alpha=0.01,
         label='TransactionAmt-train',
         title='Train and test Transaction Amounts by Time (TransactionDT)',
         ylim=(0, 5000),
         color='orange',
         figsize=(15, 5),
         ax=ax)
plt.show()

print('  {:.4f}% of Transactions that are fraud in train '.format(train_data['isFraud'].mean() * 100))
plt.figure(4)
train_data['TransactionAmt'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribution of Log Transaction Amt')
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))
train_data.loc[train_data['isFraud'] == 1] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Fraud',
          color=color_pal[1],
          xlim=(-3, 10),
         ax= ax1)
train_data.loc[train_data['isFraud'] == 0] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Not Fraud',
          color=color_pal[2],
          xlim=(-3, 10),
         ax=ax2)
train_data.loc[train_data['isFraud'] == 1] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Fraud',
          color=color_pal[1],
         ax= ax3)
train_data.loc[train_data['isFraud'] == 0] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Not Fraud',
          color=color_pal[2],
         ax=ax4)
plt.show()
print('Mean transaction amt for fraud is {:.4f}'.format(train_data.loc[train_data['isFraud'] == 1]['TransactionAmt'].mean()))
print('Mean transaction amt for non-fraud is {:.4f}'.format(train_data.loc[train_data['isFraud'] == 0]['TransactionAmt'].mean()))
plt.figure(6)

my_cmap = cm.get_cmap('jet')

train_data.groupby('ProductCD') \
    ['TransactionID'].count() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(15, 3),
          color=[plt.cm.Paired(np.arange(len(train_data.groupby('ProductCD'))))],
         title='Count of Observations by ProductCD')
plt.show()

plt.figure(7)
train_data.groupby('ProductCD')['isFraud'] \
    .mean() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(15, 3),
          color=[plt.cm.Paired(np.arange(len(train_data.groupby('ProductCD'))))],
         title='Percentage of Fraud by ProductCD')

plt.show()

#

plt.figure(9)
fig, ax = plt.subplots(1, 2, figsize=(20,5))

sns.countplot(x="ProductCD", ax=ax[0], hue = "isFraud", data=train_data)
ax[0].set_title('ProductCD train', fontsize=14)
sns.countplot(x="ProductCD", ax=ax[1], data=test_data)
ax[1].set_title('ProductCD test', fontsize=14)
plt.show()

card_cols = [c for c in train_data.columns if 'card' in c]
train_data[card_cols].head()

cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for i in cards:
    print ("Unique ",i, " = ",train_data[i].nunique())

#
#
#
#
plt.figure(11)
color_idx = 0
for c in card_cols:
    if train_data[c].dtype in ['float64','int64']:
        train_data[c].plot(kind='hist',
                                      title=c,
                                      bins=50,
                                      figsize=(15, 2),
                                      color=color_pal[color_idx])
    color_idx += 1
    plt.show()

plt.figure(12)
train_transaction_fr = train_data.loc[train_data['isFraud'] == 1]
train_transaction_nofr = train_data.loc[train_data['isFraud'] == 0]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_transaction_fr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax1, title='Count of card4 fraud')
train_transaction_nofr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax2, title='Count of card4 non-fraud')
train_transaction_fr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax3, title='Count of card6 fraud')
train_transaction_nofr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax4, title='Count of card6 non-fraud')
plt.show()
#
#
plt.figure(14)
ax = sns.countplot(x="DeviceType", data=train_data)
ax.set_title('DeviceType', fontsize=14)
plt.show()
#
#

plt.figure(15)
print ("Unique Devices = ",train_data['DeviceInfo'].nunique())
train_data['DeviceInfo'].value_counts().head()

cards = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for i in cards:
    print ("Unique ",i, " = ",train_data[i].nunique())

# cards = train_data.iloc[:,4:7].columns
# #
# # plt.figure(figsize=(18,8*4))
# # gs = gridspec.GridSpec(8, 4)
# # for i, cn in enumerate(cards):
# #     ax = plt.subplot(gs[i])
# #     sns.distplot(train_data.loc[train_data['isFraud'] == 1][cn], bins=50)
# #     sns.distplot(train_data.loc[train_data['isFraud'] == 0][cn], bins=50)
# #     ax.set_xlabel('')
# #     ax.set_title('feature: ' + str(cn))
# # plt.show()


fig, ax = plt.subplots(1, 4, figsize=(25,5))

sns.countplot(x="card4", ax=ax[0], data=train_data.loc[train_data['isFraud'] == 0])
ax[0].set_title('card4 isFraud=0', fontsize=14)
sns.countplot(x="card4", ax=ax[1], data=train_data.loc[train_data['isFraud'] == 1])
ax[1].set_title('card4 isFraud=1', fontsize=14)
sns.countplot(x="card6", ax=ax[2], data=train_data.loc[train_data['isFraud'] == 0])
ax[2].set_title('card6 isFraud=0', fontsize=14)
sns.countplot(x="card6", ax=ax[3], data=train_data.loc[train_data['isFraud'] == 1])
ax[3].set_title('card6 isFraud=1', fontsize=14)
plt.show()


print(' addr1 - has {} NA values'.format(train_data['addr1'].isnull().sum()))
print(' addr2 - has {} NA values'.format(train_data['addr2'].isnull().sum()))

plt.figure(20)

train_data['dist1'].plot(kind='hist',
                                bins=5000,
                                figsize=(15, 2),
                                title='dist1 distribution',
                                color=color_pal[1],
                                logx=True)
plt.show()
plt.figure(19)
train_data['dist2'].plot(kind='hist',
                                bins=5000,
                                figsize=(15, 2),
                                title='dist2 distribution',
                                color=color_pal[1],
                                logx=True)
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="P_emaildomain", ax=ax[0], data=train_data)
ax[0].set_title('P_emaildomain', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[1], data=train_data.loc[train_data['isFraud'] == 1])
ax[1].set_title('P_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="P_emaildomain", ax=ax[2], data=train_data.loc[train_data['isFraud'] == 0])
ax[2].set_title('P_emaildomain isFraud = 0', fontsize=14)
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(32,10))

sns.countplot(y="R_emaildomain", ax=ax[0], data=train_data)
ax[0].set_title('R_emaildomain', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[1], data=train_data.loc[train_data['isFraud'] == 1])
ax[1].set_title('R_emaildomain isFraud = 1', fontsize=14)
sns.countplot(y="R_emaildomain", ax=ax[2], data=train_data.loc[train_data['isFraud'] == 0])
ax[2].set_title('R_emaildomain isFraud = 0', fontsize=14)
plt.show()



m_cols = [c for c in train_data if c[0] == 'M']
train_data[m_cols].head()

(train_data[m_cols] == 'T').sum().plot(kind='bar',
                                              title='Count of T by M column',
                                              figsize=(15, 2),
                                              color=color_pal[3])
plt.show()
(train_data[m_cols] == 'F').sum().plot(kind='bar',
                                              title='Count of F by M column',
                                              figsize=(15, 2),
                                              color=color_pal[4])
plt.show()
(train_data[m_cols].isnull()).sum().plot(kind='bar',
                                              title='Count of NaN by M column',
                                              figsize=(15, 2),
                                              color=color_pal[0])
plt.show()


train_data.groupby('M4')['TransactionID'] \
    .count() \
    .plot(kind='bar',
          title='Count of values for M4',
          figsize=(15, 3))
plt.show()


#model = LogisticRegression()
estimator =  ()
# create the RFE model and select 3 attributes
selector= RFE(estimator, 0.1)
y_target =list

selector = selector.fit(train_data,y_target)
# summarize the selection of the attributes
print(selector.support_)
print(selector.ranking_)


x=0

