# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#acquire data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


train_df.info()

#Data description in percentiles
train_df.describe()

#Distribution of categorical features
train_df.describe(include=['O'])

#Observe correlation continuous numerical features
#Use
contnum=['LotArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','WoodDeckSF','OpenPorchSF','GarageArea']
for feat in contnum:
    train_df.plot(kind='scatter',x=feat,y='SalePrice')

#Nouse
contnum_nouse=['LotFrontage','BsmtFinSF2','LowQualFinSF','PoolArea','MasVnrArea']

#Check for Nan
train_df[contnum].isnull().sum()
test_df[contnum].isnull().sum()

#Complete Nan for continuous feats with mean
for dataset in combine:
    for feat in contnum:
        dataset[feat].fillna(value=dataset[feat].mean(),inplace=True)





