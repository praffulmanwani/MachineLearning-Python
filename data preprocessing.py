import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:].values
#now handle missing valuse using mean
from sklearn.preprocessing import Imputer
imputer =  Imputer(missing_values= "NaN" , strategy = "mean" , axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
#now we handle the catogorise valible and encode we have  two country purchase
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
lb_x = LabelEncoder()
x[:,0] = lb_x.fit_transform(x[:,0]) 
#now the data of country in 0 1 2 but ml algo thinks 0 is lt 1 remove this we use one hot encoder
ohe = OneHotEncoder(categorical_features=[0])
x = ohe.fit_transform(x).toarray()
# now wfor column purchased
lb_y = LabelEncoder()
y = lb_y.fit_transform(y)
#spitting data set into training set and test set
from sklearn.model_selection import train_test_split
xt , xtest , yt, ytest = train_test_split(x,y,test_size = 0.2 , random_state = 0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xt = sc_x.fit_transform(xt)
xtest = sc_x.transform(xtest)