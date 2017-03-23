import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.linear_model import LinearRegression
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import sys
import math

xtimesw = np.loadtxt('xtimes.txt')
xtimesm = (xtimesw[:,0] + xtimesw[:,1] ) * 0.5

layers = 3
data_1_lyr = np.loadtxt('training_data/training_0%d_layer.csv'%(layers),skiprows=1,delimiter=',',usecols=range(1,21))
Xtrain = data_1_lyr[0:9000,0:(2*layers)]
Ytrain = np.log(data_1_lyr[0:9000,(2*layers):(2*layers+18)])
Xtest = data_1_lyr[9000:,0:(2*layers)]
Ytest = np.log(data_1_lyr[9000:,(2*layers):(2*layers+18)])


def main():
        #clsfr = MultiOutputRegressor(linear_model.LinearRegression(),n_jobs=-1)
        clsfr = Pipeline([('poly',PolynomialFeatures(degree=3)),
                          ('linear', LinearRegression(fit_intercept=False))])

        clsfr.fit(Xtrain, Ytrain)

        print "Score: " + str(clsfr.score(Xtest,Ytest))

        #Ypred_class_prob = clsfr.predict_proba(Xtest)
        Ypred = clsfr.predict(Xtest)

        #f, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(3,2)
        f, ((ax1)) = plt.subplots(1,1)

        #ax1.plot(Xtest, Ypred, color='darkorange',
        #         lw=2)
        bins = np.arange(0,1,0.01)
        #residuals = np.divide(Ytest-Ypred,Ytest)
        residuals = np.divide(Ytest-Ypred,Ytest)
        resim = np.zeros((bins.size - 1,Ytest.shape[1]))
        for tw in range(Ytest.shape[1]):
                (h,be) = np.histogram(np.log1p(np.absolute(residuals[:,tw])),bins)
                resim[:,tw] = h
        ax1.imshow(resim,interpolation=None,aspect='auto')
        #ax1.scatter(np.tile(np.arange(Ytest.shape[1]),Ytest.shape[0]), np.ravel(np.divide(Ytest-Ypred,Ytest)), color='navy')
        #ax1.scatter(np.tile(xtimesm,Ytest.shape[0]), np.ravel(np.divide(Ytest-Ypred,Ytest)), color='navy')
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Response Residuals')
        ax1.set_title('Residuals of Predicted Responses')
        ax1.legend(loc="lower right")
        plt.show()

if __name__ == '__main__':
	main()
