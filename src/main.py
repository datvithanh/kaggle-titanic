import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from preprocess import preprocess
import logistic_regression

train = pd.read_csv('../data/train.csv')
holdout = pd.read_csv('../data/test.csv')

all = train.drop(["Survived"], axis = 1).append(holdout)

all = preprocess(all)

# columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Family_categories_Singleton', 'Family_categories_Small', 'Family_categories_Large', 'Sex_female', 'Sex_male', 'Age_categories_Missing', 'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 'Fare_categories_0-12', 'Fare_categories_12-50', 'Fare_categories_50-100', 'Fare_categories_100+', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_U', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'Ticket_A', 'Ticket_A4', 'Ticket_A5', 'Ticket_AQ3', 'Ticket_AQ4', 'Ticket_C', 'Ticket_CA', 'Ticket_FC', 'Ticket_FCC', 'Ticket_LP', 'Ticket_PC', 'Ticket_PP', 'Ticket_SC', 'Ticket_SCA3', 'Ticket_SCA4', 'Ticket_SCAH', 'Ticket_SCPARIS', 'Ticket_SCParis', 'Ticket_SOC', 'Ticket_SOPP', 'Ticket_SOTONO2', 'Ticket_SOTONOQ', 'Ticket_STONO', 'Ticket_STONO2', 'Ticket_STONOQ', 'Ticket_U', 'Ticket_WC', 'Ticket_WEP']
all_X = all.iloc[:891]
all_y = train['Survived']
holdout_X = all.iloc[891:]


holdout_predictions = logistic_regression.model(all_X, all_y, holdout_X)


# holdout_ids = holdout["PassengerId"]
# submission_df = {"PassengerId": holdout_ids,
#                  "Survived": holdout_predictions}
# submission = pd.DataFrame(submission_df)

# submission.to_csv("../data/submission.csv", index=False)