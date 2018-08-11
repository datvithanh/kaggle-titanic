from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def model(all_X, all_y, holdout_X):
    
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.1, random_state=0)

    forest = RandomForestClassifier(n_estimators=100,
                                    criterion='gini',
                                    max_depth=5,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
                                    random_state=0)
    forest.fit(train_X, train_y)
    print("Random Forest score: {0:.2}".format(forest.score(test_X, test_y)))

    clf = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
    clf.fit(all_X, all_y)
    prediction = clf.predict(holdout_X)
    return prediction
