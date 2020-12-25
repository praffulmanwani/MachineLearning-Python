import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values 

#dont require trainset and test set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#use polynomial reg

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 3)
x_poly = poly_reg.fit_transform(x)
reg_new = LinearRegression()
reg_new.fit(x_poly,y)
#scatter plot
pet.scatter(x,y,color = "red")
pet.plot(x , regressor.predict(x), color = "blue")
pet.show()
#scatter olt for polynomial
pet.scatter(x,y,color = "red")
pet.plot(x , reg_new.predict(poly_reg.fit_transform(x)), color = "blue")
pet.show()


