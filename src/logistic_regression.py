import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def model(all_X, all_y, holdout_X):
    #test

    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.1, random_state=0)

    lr = LogisticRegression()

    lr.fit(train_X, train_y)
    predictions = lr.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)
    print('Test accuracy:', accuracy)

    #cross validation
    scores = cross_val_score(lr, all_X, all_y, cv=10)

    accuracy = np.mean(scores)

    print('Cross validation scores:', scores)

    print('Cross validation accuracy', accuracy)

    #predict
    lr = LogisticRegression()
    lr.fit(all_X, all_y)
    holdout_predictions = lr.predict(holdout_X)
    return holdout_predictions