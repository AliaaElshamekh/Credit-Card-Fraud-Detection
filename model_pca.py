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
import pickle
import time
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('fraud_sample_new3.csv')
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



X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
max_f1 = 0

#
X_train = StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform((X_test))
pca = PCA(copy=True, iterated_power='auto', n_components=120, random_state=None,
            svd_solver='auto', tol=0.0, whiten=False)
# lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
#               solver='svd', store_covariance=True, tol=0.0001)
#
# X_train = lda.fit_transform(X_train,len(labels[0:len(X_train)]))
# X_test = lda.transform(X_test,labels.reshape(len(labels[len(X_train):len(labels)])))

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
feat_idx = list(range(X_train.shape[1]))
for i in range(100):
    trees_1 = int(np.random.random()*150) + 5
    trees_2 = int(np.random.random() * 150) + 5
    depth_1 = int(np.random.random()*20) + 2
    h3 = int(np.random.random() * 50) + 5
    n_neighbors =int(np.random.random() * 25) + 5
    #clf = MLPClassifier(hidden_layer_sizes=h3, activation='relu', random_state=0,max_iter=10000)
    #clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=20), max_samples=0.5, max_features=0.5,n_estimators=trees_2)
    #clf = BaggingClassifier(SVC(kernel="rbf", random_state=0), max_samples=0.5, max_features=0.5,n_estimators=trees_2)
    #SVC(kernel="rbf", random_state=0)
    clf = AdaBoostClassifier(n_estimators=trees_2, random_state=0)
    #clf = RandomForestClassifier(n_estimators=trees_1, max_depth=depth_1,random_state=0,class_weight = 'balanced')
    #clf = BaggingClassifier(DecisionTreeClassifier(max_depth=depth_1),max_samples = 0.75, max_features = 0.5,n_estimators=200)
    #clf = BaggingClassifier(SVC(kernel="rbf",random_state=0),max_samples = 0.5, max_features = 0.5,n_estimators=trees_2)
    #clf = SVC(kernel="rbf",random_state=0)
    clas = classify(feat_idx,X_train,y_train,X_test,y_test,clf,max_f1)
    clas.fit()
    clas.predict()
    f1,f1_train = clas.get_f1()
    if f1>max_f1:
        print(f1,f1_train)
        max_f1 = f1
        output = open('data'+str(time.time())+'.pkl', 'wb')
        pickle.dump(clas, output)
        output.close()