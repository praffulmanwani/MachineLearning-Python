import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:].values


#spitting data set into training set and test set
from sklearn.model_selection import train_test_split
xt , xtest , yt, ytest = train_test_split(x,y,test_size =1/3, random_state = 0)
"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xt = sc_x.fit_transform(xt)
xtest = sc_x.transform(xtest)
"""
#fitting simple linear regression test into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xt,yt)
 #predict the salary using xtest exerience
 y_pred = regressor.predict(xtest)
#plot graph
pet.scatter(xt,yt,color = "red")
pet.plot(xt , regressor.predict(xt) )
pet.xlabel("exp")
pet.ylabel("sal")
pet.show() 
 #plot graph of test set
pet.scatter(xtest,ytest,color = "red")
pet.plot(xt , regressor.predict(xt))
pet.xlabel("exp")
pet.ylabel("sal")
pet.show()
