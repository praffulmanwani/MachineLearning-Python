import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3:].values


#spitting data set into training set and test set
from sklearn.model_selection import train_test_split
xt , xtest , yt, ytest = train_test_split(x,y,test_size = 0.2 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xt = sc_x.fit_transform(xt)
xtest = sc_x.transform(xtest)