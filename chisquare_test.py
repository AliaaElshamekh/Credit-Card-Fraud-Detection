import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = "{0} is IMPORTANT for Prediction".format(colX)
        else:
            result = "{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        print (self.p)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)


train_dataset_url = 'fraud.csv'
train_data = pd.read_csv(train_dataset_url)

# Initialize ChiSquare Class
cT = ChiSquare(train_data)
testColumns =['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain','M1',
               'M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_32','id_28',
              'id_29','id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38','id_28','id_28',
              'id_15','id_16','id_12']
#
#
X= list()
Y= list()
for var in testColumns:
    cT.TestIndependence(colX=var,colY="isFraud")
    print (cT.p)
    print (var)
    X.append(var)
    Y.append(cT.p)

x=0

#train_data=train_data.drop(columns=testColumns,axis=1)
#
# data = train_data.to_numpy()
# clf = LinearDiscriminantAnalysis()
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# data = imp.fit_transform(data)
# clf.fit(data, data[:,1])
# LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
#               solver='svd', store_covariance=True, tol=0.0001)
# clf.coef_