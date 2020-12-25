import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values
#now we handle the catogorise valible and encode we have  two country purchase
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
lb_x = LabelEncoder()
x[:,3] = lb_x.fit_transform(x[:,3]) 
#now the data of country in 0 1 2 but ml algo thinks 0 is lt 1 remove this we use one hot encoder
ohe = OneHotEncoder(categorical_features=[3])
x = ohe.fit_transform(x).toarray()
#avoiding the dummy variable trap
x = x[:,1:]


#spitting data set into training set and test set
from sklearn.model_selection import train_test_split
xt , xtest , yt, ytest = train_test_split(x,y,test_size = 0.2 , random_state = 0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xt = sc_x.fit_transform(xt)
xtest = sc_x.transform(xtest)"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xt,yt)
y_pred = regressor.predict(xtest)
#now we use backward elimination to eliminate unnessary indepandant varibale
import statsmodels.api as sm
#now we add constant column contain only ones
x = np.append(arr = np.ones((50,1)).astype(int) , values = x , axis = 1)
x_opt=x[: , [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()
#remove 2 column because high p valuse
x_opt=x[: , [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()
#remove 1,4,5 column because high p valuse
x_opt=x[: , [0,3]]
regressor_ols = sm.OLS(endog = y , exog = x_opt).fit()
regressor_ols.summary()
