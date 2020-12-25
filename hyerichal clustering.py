#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#load data set into x varibale
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values
#find the optimal no of clusters
import scipy.cluster.hierarchy as sch
dendogram1 = sch.dendrogram(sch.linkage(x,method = "ward"))

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5)
y_pred = hc.fit_predict(x)
plt.scatter(x[y_pred == 0,0] ,x[y_pred == 0,1] , s = 100 ,c = "red", label = "careful" )
plt.scatter(x[y_pred == 1,0] ,x[y_pred == 1,1] , s = 100 ,c = "cyan", label = "standard" )
plt.scatter(x[y_pred == 2,0] ,x[y_pred == 2,1] , s = 100 ,c = "blue", label = "target")
plt.scatter(x[y_pred == 3,0] ,x[y_pred == 3,1] , s = 100 ,c = "green", label = "careless")
plt.scatter(x[y_pred == 4,0] ,x[y_pred == 4,1] , s = 100 ,c = "magenta", label = "sensible" )
plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1]  , s = 300 ,c = "yellow" )
plt.legend()