import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
from matplotlib.dates import datestr2num, num2date
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
import sys
import math


data_1_lyr = np.loadtxt('training_data/training_01_layer.csv',skiprows=1,delimiter=',',usecols=range(1,21))
Xtrain = data_1_lyr[0:1000,0:2]
Ytrain = np.log(data_1_lyr[0:1000,2:20])
Xtest = data_1_lyr[1000:,0:2]
Ytest = np.log(data_1_lyr[1000:,2:20])


if True:
	#clsfr = linear_model.LogisticRegression(C=1e5)
	#clsfr = BaggingClassifier(linear_model.LogisticRegression(C=1e2),n_estimators=100,bootstrap_features=False)
	#clsfr = BaggingClassifier(SVC(verbose=True,probability=True),n_estimators=100,bootstrap_features=False)
	#clsfr = AdaBoostClassifier(SVC(verbose=True,probability=True),n_estimators=50)
	#clsfr = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.3, max_depth=3, random_state=0)
	#clsfr = DecisionTreeClassifier(random_state=0)
	#clsfr = SVC(kernel='linear',degree=3,verbose=True,probability=True)
	#clsfr = SVC(verbose=True,probability=True)
	#clsfr = RandomForestClassifier(verbose=True)
	#clsfr = MLPClassifier(solver='lbfgs',verbose=True)
	clsfr = MultiOutputRegressor(linear_model.LinearRegression(),n_jobs=-1)

	clsfr.fit(Xtrain, Ytrain)

	print "Score: " + str(clsfr.score(Xtest,Ytest))

	#Ypred_class_prob = clsfr.predict_proba(Xtest)
	Ypred_classification = clsfr.predict(Xtest)

	f, ax = plt.subplots(1,1)
        
        print np.shape(Ypred_classification)

        ax.scatter(Ytest.ravel(), Ypred_classification.ravel())
        plt.savefig('a.pdf')
