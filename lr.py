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

	#f, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2)
	f, ((ax1,ax2)) = plt.subplots(1,2)

	fpr,tpr,thresholds = metrics.roc_curve(Ytest,Ypred_1,pos_label=1,drop_intermediate=False)
	roc_auc = metrics.auc(fpr,tpr)

	fprc,tprc,thresholdsc = metrics.roc_curve(Ytest,Ypred_classification,pos_label=1,drop_intermediate=False)
	roc_aucc = metrics.auc(fprc,tprc)

	lw = 2

	ax1.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	ax1.set_xlim([0.0, 1.0])
	ax1.set_ylim([0.0, 1.05])
	ax1.set_xlabel('False Positive Rate')
	ax1.set_ylabel('True Positive Rate')
	ax1.set_title('Receiver operating characteristic test data')
	ax1.legend(loc="lower right")
