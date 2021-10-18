import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from classify import classify
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import  RFECV

from sklearn.naive_bayes import GaussianNB
import pickle
import time
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
df = pd.read_csv('fraud_sample_new3.csv')

test_dataset_url ='fraud_test_sample.csv'
test_data = pd.read_csv(test_dataset_url)

cols_dt  = {}
cols = []
for c,dt in zip(df.columns,df.dtypes):
    cols_dt[c] = dt
    cols.append(c)
cols = cols[2:]
data = df.to_numpy()
labels = data[:,1].astype(int)
data = data[:,2:]
shp = data.shape
for i in range(shp[1]):
    dt = cols_dt[cols[i]]
    data[:,i] = data[:,i].astype(dt)

for i in range(shp[1]):
    tmp = data[:,i]
    tmp = tmp.reshape(-1,1 )
    try:
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        tmp = imp.fit_transform(tmp)
        x = 0

    except:
        imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        tmp = imp.fit_transform(tmp)
        x = 0
        enc = LabelEncoder()
        tmp = enc.fit_transform(tmp)
    tmp = tmp.reshape(len(tmp,))
    data[:, i] = tmp


# remove mean and scale to unit variance
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
#X_train, y_train = SMOTE().fit_resample(X_train, y_train)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
max_f1 = 0
for i in range(200):
    trees_1 = int(np.random.random()*150) + 5
    depth_1 = int(np.random.random()*20) + 2
    trees_2 = int(np.random.random() * 150) + 5
    depth_2 = int(np.random.random() * 20) + 2
    h2 = int(np.random.random() * 150) + 5
    h3 = int(np.random.random() * 50) + 5
    refclf = RandomForestClassifier(n_estimators=trees_1, max_depth=depth_1,
                                   random_state=0,class_weight = 'balanced')
    clf = RandomForestClassifier(n_estimators=trees_2, max_depth=depth_2,
                                  random_state=0,class_weight = 'balanced')
    #refclf = AdaBoostClassifier(n_estimators=100, random_state=0)

    #refclf = SVC(kernel="linear",random_state=0)

    #clf = MLPClassifier(hidden_layer_sizes = h3,activation='relu', random_state=0)
   # clf = BaggingClassifier(KNeighborsClassifier(),max_samples = 0.5, max_features = 0.5)
    #clf = AdaBoostClassifier(n_estimators=trees_2, random_state=0)
    rfecv = RFECV(estimator=refclf, step=0.2, cv=3, scoring='f1',n_jobs =-1 )

    #X_train = preprocessing.scale(X_train)
    #y_train = preprocessing.scale(y_train)
    rfecv.fit(X_train, y_train)
    feat_idx = rfecv.support_



    clas = classify(feat_idx,X_train,y_train,X_test,y_test,clf,max_f1,rfecv)
    clas.fit()
    clas.predict()
    f1,f1_train = clas.get_f1()
    if f1>max_f1:
        print(f1,f1_train)
        max_f1 = f1
        output = open('data'+str(time.time())+'.pkl', 'wb')
        pickle.dump(clas, output)
        output.close()

        '''
         file = open('data1572197365.9462965.pkl', 'rb')

        # dump information to that file
        dddd = pickle.load(file)

        # close the file
        file.close()
        '''

x = 0

