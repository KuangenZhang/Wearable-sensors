# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:31:54 2019

@author: kuangen
"""
#%% LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score
import numpy as np
acc_s_LDA = np.zeros(10)
for i in range(10):
	# change the load_data() function to your own data loading function
    x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test = load_data()
    clf = LDA()
    clf.fit(x_s_train, y_s_train)
    y_s_test_pred = clf.predict(x_s_test)
    acc = accuracy_score(y_s_test, y_s_test_pred)
    print("LDA: test accuracy: %.2f%%" % acc)
    acc_s_LDA[i] = acc

print ('Mean of test acc:', np.mean(acc_s_LDA))

#%% SVM
from sklearn import svm
acc_s_SVM = np.zeros(10)
for i in range(10):
	# change the load_data() function to your own data loading function
    x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test = load_data()
    clf = svm.LinearSVC(max_iter=5000, verbose=1, C=10)
    clf.fit(x_s_train, y_s_train)
    y_s_test_pred = clf.predict(x_s_test)
    acc = accuracy_score(y_s_test, y_s_test_pred)
    print("SVM: test accuracy: %.2f%%" % acc)
    acc_s_SVM[i] = acc
   
print ('Mean of test acc:', np.mean(acc_s_SVM))

#%% ANN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import FileIO
acc_s_ANN = np.zeros(10)
for i in range(10): 
	# change the load_data() function to your own data loading function
    x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test = load_data()
    
    clf = MLPClassifier(solver='sgd', activation='tanh',learning_rate='adaptive',
                        learning_rate_init=0.1,hidden_layer_sizes=(10,),
                        max_iter = 2000)
    clf.fit(x_s_train, y_s_train)
    y_s_test_pred = clf.predict(x_s_test)
    acc = accuracy_score(y_s_test, y_s_test_pred)
    print("ANN: test accuracy: %.2f%%" % acc)
    acc_s_ANN[i] = acc

print ('Mean of test acc:', np.mean(acc_s_ANN))