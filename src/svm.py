import pandas as pd 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split


def model(all_X, all_y, holdout_X):

    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.1, random_state=0)

    clf = svm.SVC(kernel = 'rbf')
    clf.fit(train_X, train_y)
    predicted = clf.predict(test_X)
    report = classification_report(test_y, predicted)
    accuracy = clf.score(test_X, test_y)
    print('Accuracy:', accuracy*100)
    print('Report:', report)
    
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(all_X, all_y)
    holdout_predictions = clf.predict(holdout_X)
    return holdout_predictions
