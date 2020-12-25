#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#load data set into x varibale
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values
#find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans =KMeans(n_clusters= i,random_state= 0 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
#now we find optimal no of clusters = 5
kmeans =KMeans(n_clusters= 5,random_state= 0 )
kmeans.fit(x)
#find the predicted the value
y_pred = kmeans.predict(x)
#now we visualize the data
plt.scatter(x[y_pred == 0,0] ,x[y_pred == 0,1] , s = 100 ,c = "red", label = "careful" )
plt.scatter(x[y_pred == 1,0] ,x[y_pred == 1,1] , s = 100 ,c = "cyan", label = "standard" )
plt.scatter(x[y_pred == 2,0] ,x[y_pred == 2,1] , s = 100 ,c = "blue", label = "target")
plt.scatter(x[y_pred == 3,0] ,x[y_pred == 3,1] , s = 100 ,c = "green", label = "careless")
plt.scatter(x[y_pred == 4,0] ,x[y_pred == 4,1] , s = 100 ,c = "magenta", label = "sensible" )
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1]  , s = 300 ,c = "yellow" )
plt.legend()

