# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:08:28 2021

@author: prabhu
"""
#OBJECTIVE- TO PREDICT SURVIVAL BASED ON VARIOUS INDEPENDENT FEATURES
##importing datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##getting the dataset
data=pd.read_csv("titanic.csv")
data.shape
pd.pandas.set_option("max_columns",None)
data.head()
#missing values
data_nan=[feature for feature in data.columns if data[feature].isnull() .sum()>1]
data_nan
##wwe need to print the % of nan values of all features
for feature in data_nan:
    print(feature,"% of nan values:",np.round(data[feature].isnull().mean(),4))
##about 77% of data is nan in cabin section
##we will deal with that in feature engineering
##from the above dataset lets find a correlation between various features
data.corr()
sns.heatmap(data.corr())
##parch and sibsp is very much correlated
##here in this dataset we donot need name passenger id so lets drop the columns
dataset=data.drop(["PassengerId","Name"],axis=1)
dataset.head()
##lets check for columns having categorical values
data_cat=[feature for feature in data.columns if data[feature].dtypes=="O"]
data_cat
##we have got the categorical values
##need to encode them in feature engineering
dataset.to_csv("dataset.csv")
##now that we got our values of nans and categorical values 
##we need to clean the data
##importing datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##getting the dataset
data=pd.read_csv("dataset.csv")
data.head()
dataset=data.drop(["Unnamed: 0"],axis=1)
dataset.head()
##fixing missing values
data_nan=[feature for feature in data.columns if data[feature].isnull() .sum()>1]
data_nan
for feature in data_nan:
    print(feature,"% of nan values:",np.round(data[feature].isnull().mean(),4))
sns.heatmap(dataset.isnull())
##from the heatmap we can see that there is a lot of null values 
##we can either drop the column
##or fit the average values in place of null values
##for age
mean_value=dataset["Age"].mean()
dataset["Age"].fillna(value=mean_value,inplace=True)
dataset=dataset.drop(["Cabin"],axis=1)
##no more null values
##the ticket value is irrelant so we can drop it
dataset1=dataset.drop(["Ticket"],axis=1)
dataset.head()
dat_cat=[feature for feature in dataset.columns if dataset[feature].dtypes=="O"]
dat_cat
##lets fix the categorical values
sex=pd.get_dummies(dataset["Sex"],drop_first=True)
embarked=pd.get_dummies(dataset["Embarked"],drop_first=True)
dataset=pd.concat([dataset,sex,embarked],axis=1)
dataset.head()
dataset=dataset.drop(["Sex","Ticket","Embarked"],axis=1)
dataset.head()
dataset.to_csv("data.csv")
##importing datasets
import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv("data.csv")
data.head()
data.head()
sns.pairplot(data)
##from this pairplot we can see parch and sibsp as highly correlated so we can take any one of the two
##lets drop other columns except this
data=data.drop(["SibSp","Fare"],axis=1)
data.head()
data=data.drop(["Unnamed: 0"],axis=1)
data.head()
##the data is ready now lets separate the dependent vatiables and independent variables
y=data["Survived"]
y.head()#independent variable
x=data.drop(["Survived"],axis=1)
x.head()##dependent variables
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=42)
##implementing random forest classfier
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score
#Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
##saving model to disk using pickle
pickle.dump(regressor,open("model.pkl","wb"))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))