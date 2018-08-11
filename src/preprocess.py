import pandas as pd
import numpy as np



def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

#binning age
def process_age(df,cut_points = [-1, 0, 5, 12, 18, 35, 60, 100], label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

#categorize social position
def process_name(df):
    title_dict = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }

    titles = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = titles.map(title_dict)
    return df

#categorize cabin type
def process_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Cabin'] = df['Cabin'].str[0]
    return df

#categorize ticket
def clean(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    return ticket[0] if len(ticket) > 0 else 'U'

def process_ticket(df):
    df['Ticket'] = df['Ticket'].map(lambda ticket:clean(ticket))
    return df

#categorize family
def process_family(df, cut_points = [0.5, 1.5, 4.5, 11.5], label_names = ['Singleton', 'Small', 'Large']):
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df["Family_categories"] = pd.cut(df["FamilySize"],cut_points,labels=label_names)

    return df

#categorize fare
def process_fare(df, cut_points = [0, 12, 50, 100, 1000], label_names = ["0-12","12-50","50-100","100+"]):
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df
def preprocess(df):
    df = process_age(df)
    df = process_name(df)
    df = process_cabin(df)
    df = process_ticket(df)
    df = process_family(df)
    df = process_fare(df)

    for column in ["Pclass", "Family_categories", "Sex","Age_categories", "Fare_categories","Cabin", "Embarked", "Title", "Ticket"]:
        df = create_dummies(df,column)
    
    df = df.drop(["PassengerId", "FamilySize", "Family_categories", "Fare_categories", "Name", "Sex",  "SibSp", "Parch", "Fare", "Ticket", "Cabin", "Embarked", "Age", "Pclass", "Title", "Age_categories"], axis = 1)
    
    return df 