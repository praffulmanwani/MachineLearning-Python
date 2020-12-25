import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values 

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

#fitting regression model to data set


#predict the value
y_pred = regressor.predict(x)

#scatter polt
pet.scatter(x,y,color = "red")
pet.plot(x ,regressor.predict(x), color = "blue")
pet.show()
#scatter plot for higher resolution
x_grid = np.arange(min(x),max(x),0.1 )
x_grid = x_grid.reshape((len(x_grid),1))
pet.scatter(x,y,color = "red")
pet.plot(x_grid ,regressor.predict(x_grid), color = "blue")
pet.show()



