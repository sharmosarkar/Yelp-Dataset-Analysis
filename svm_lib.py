'''
    This file implements SVM using various kernels from the sklearn package 
'''

import numpy as np
import scipy.io
from numpy import loadtxt, where
from sklearn import svm

DATA_LOCATION = 'Z:/ML/HW-2/code/data.mat'

def get_input_data():
    data = scipy.io.loadmat(DATA_LOCATION)
    return data['X_trn'], data['Y_trn'], data['X_tst'], data['Y_tst']

def run_SVM(X_train_data,Y_train_data,X_test_data,kernel_type='linear'):
    clf = svm.SVC(kernel=kernel_type, C=1.0)
    clf.fit(X_train_data, Y_train_data.ravel())
    predictions = clf.predict(X_test_data)
    return (predictions,"ss")


def run_SVM1(X_train_data,Y_train_data,kernel_type):
    print("enters svm")
    clf = svm.SVC(kernel=kernel_type, C=1.0)
    clf.fit(X_train_data, Y_train_data.ravel())
    predictions = clf.predict(X_train_data[0])
    print("p is", predictions)
    result = calc_accuracy(predictions,Y_train_data)
    print("Prediction accuracy for kernel type as ", kernel_type, result)

def calc_accuracy(pred_lst,gold_lst):
    hits = 0.0
    for pred,gold in zip(pred_lst,gold_lst):
        if gold == pred:
            hits += 1
    return (hits*1.0 / len(pred_lst))*100


def train(X_train_data, Y_train_data, kernel):
    # print(X_train_data,"hello")
    run_SVM1(X_train_data.as_matrix(),Y_train_data.as_matrix().astype(int),kernel)


def initialize_SVM (X_train_data, Y_train_data, X_test_data=None, Y_test_data=None):
    train(X_train_data, Y_train_data,'linear')
    train(X_train_data, Y_train_data,'poly')
    train(X_train_data, Y_train_data,'sigmoid')
    train(X_train_data, Y_train_data,'rbf')



def main():
    X_train_data, Y_train_data, X_test_data, Y_test_data = get_input_data()
    kernel_type_lst = ['linear','poly','sigmoid','rbf']
    for kernel_type in kernel_type_lst:
        print('Accuracy using ',kernel_type,' kernel = ',\
            calc_accuracy(run_SVM(X_train_data,Y_train_data,X_test_data,kernel_type),Y_test_data) , '%')

if __name__ == "__main__":
    main()
^^^()()^^^