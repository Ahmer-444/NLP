import scipy.io
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

from sklearn.externals import joblib

# spamTrain.mat & spamTest.mat can be generated using NLP_Preprocessing.py
# I am using the spam dataset created by Andre Ng for Machine learning Course on Coursera
mat = scipy.io.loadmat('spamTrain.mat')
X = mat['X']
Y = mat['y']

# Naive bayes classifier
model1 = MultinomialNB()

# SVM
model2 = LinearSVC()
# Model Fitting
model1.fit(X,Y)
model2.fit(X,Y)

joblib.dump(model1, 'spamNaivemodel.pkl')
joblib.dump(model2, 'spamSVMmodel.pkl')

model1 = joblib.load('spamNaivemodel.pkl')
model2 = joblib.load('spamSVMmodel.pkl')

# Calculate training accuracy --- Naive bayes classifier

predictY = model1.predict(X)
predictY = np.reshape(predictY , (predictY.shape[0], 1))
train_accuracy = (np.sum(Y == predictY) / float(Y.shape[0]))* 100
print (" 'Training Accuracy: " + str(train_accuracy) + '%')

# Calculate training accuracy --- SVM
predictY = model2.predict(X)
predictY = np.reshape(predictY , (predictY.shape[0], 1))
train_accuracy = (np.sum(Y == predictY) / float(Y.shape[0]))* 100
print (" 'Training Accuracy: " + str(train_accuracy) + '%')


print ("------------- Training Stage Done ---------------")
test_mat = scipy.io.loadmat('spamTest.mat')
Xtest = test_mat['Xtest']
Ytest = test_mat['ytest']


# Calculate testing accuracy --- Naive bayes classifier
predictY_test = model1.predict(Xtest)
predictY_test = np.reshape(predictY_test , (predictY_test.shape[0], 1))
test_accuracy = (np.sum(Ytest == predictY_test) / float(Ytest.shape[0]))* 100
print (" 'Testing Accuracy: " + str(test_accuracy) + '%')


# Calculate testing accuracy --- SVM
predictY_test = model2.predict(Xtest)
predictY_test = np.reshape(predictY_test , (predictY_test.shape[0], 1))
test_accuracy = (np.sum(Ytest == predictY_test) / float(Ytest.shape[0]))* 100
print (" 'Testing Accuracy: " + str(test_accuracy) + '%')


'''
print confusion_matrix(YY,result1)
print confusion_matrix(YY,result2)
'''

print ("------------- Testing Done ---------------")

