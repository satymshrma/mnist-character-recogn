#%%
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
import time
import sys
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
#%%
DeprecationWarning('ignore')
warnings.filterwarnings('ignore',message='Unwanted error')

#%%
os.getcwd()

#%%
os.chdir('D:/Machine Learning/mnist-in-csv')

#%%
os.listdir()

#%%
df=pd.read_csv('mnist_train.csv')
#%%
image=df.iloc[0:1,1:]
#%%
plt.imshow(image.values.reshape(28,28))

#%%
image=df.iloc[3:4,1:]
plt.imshow(image.values.reshape(28,28))

#%%
train,test=train_test_split(df,test_size=0.3,random_state=25)
#%%
def x_y(arg):
    x=arg.drop(['label'],axis=1)
    y=arg['label']
    return x,y

#%%
x_train,y_train=x_y(train)

x_test,y_test=x_y(test)


#%%
rf=rfc(n_estimators=18)
rf.fit(x_train,y_train)

#%%
score=accuracy_score(y_train,rf.predict(x_train))

#%%
score

#%%import open cv -> use imread -> read image as csv
#%%work on fashion set