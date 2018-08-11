import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from preprocess import preprocess
import logistic_regression
import svm

train = pd.read_csv('../data/train.csv')
holdout = pd.read_csv('../data/test.csv')

all = train.drop(["Survived"], axis = 1).append(holdout)

all = preprocess(all)

all_X = all.iloc[:891]
all_y = train['Survived']
holdout_X = all.iloc[891:]

# holdout_predictions = logistic_regression.model(all_X, all_y, holdout_X)
holdout_predictions = svm.model(all_X, all_y, holdout_X)

holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("../data/submission.csv", index=False)