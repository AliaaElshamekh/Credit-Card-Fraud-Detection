from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,classification_report,confusion_matrix
import numpy as np
class classify:
    def __init__(self,feat_idx,x_train,y_train,x_test,y_test,clf,max_f1,rfecv=None):
        self.rfecv = rfecv
        self.feat_idx= feat_idx
        self.x_train = x_train
        self.y_train = np.ravel(y_train)
        self.x_test = x_test
        self.y_test= np.ravel(y_test)
        self.clf = clf
        self.max_f1 =max_f1
        self.pred = None
        self.pred_train = None
        self.fscore = 0
    def fit(self):

        self.clf.fit(self.x_train[:,self.feat_idx],self.y_train)
    def predict(self):
        self.pred = self.clf.predict(self.x_test[:,self.feat_idx])
        self.pred_train = self.clf.predict(self.x_train[:, self.feat_idx])
    def get_f1(self):
        self.fscore = f1_score(self.y_test,self.pred)
        f_train = f1_score(self.y_train,self.pred_train)
        return self.fscore,f_train
    def get_accuracy_score(self):
        self.acccuracy_score = accuracy_score(self.y_test,self.pred)
    def get_recall_score(self):
        self_recall_score = recall_score(self.y_test,self.pred)
    def get_classfication_report(self):
        self.class_report = classification_report(self.y_test,self.pred,target_names=['fraud','non-fraud'])
    def get_confusion_matrix(self):
        self.confusion_matrix = confusion_matrix(self.y_test,self.pred,labels=["fraud","non-fraud"])