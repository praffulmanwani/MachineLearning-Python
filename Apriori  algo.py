#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#load data set into x varibale
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header = None)
trans = []
for i in range(0,7501):
    trans.append([str(dataset.values[i,j]) for j in range(0,20)])
#import apyori
from apyori import apriori
z = apriori(trans,min_support = 0.003 , min_confidence = 0.2 , min_lift = 3 , min_length = 2)     

zen = list(z)
