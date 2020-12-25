import numpy as np
from matplotlib import pyplot as pet
import pandas as pd
#import dataset using pandas library
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:,-1].values


#spitting data set into training set and test set
from sklearn.model_selection import train_test_split
xt , xtest , yt, ytest = train_test_split(x,y,test_size = 0.25 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xt = sc_x.fit_transform(xt)
xtest = sc_x.transform(xtest)

#fit the data in the logistic regressor 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(xt,yt)
#predict the value using of y using xtest
y_pred = classifier.predict(xtest)
#for check the model efficency we construct a confusion matrix
#confusion matrix take org and pred data and create a matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)
