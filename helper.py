import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import pickle

def regression(data,predict):
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    
    # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
    best = 0
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)
        print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        
    return linear,best

