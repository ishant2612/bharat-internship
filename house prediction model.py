! pip install numpy
! pip install pandas
! pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\bharat intern\\house.csv - house.csv.csv")

! pip install scikit-learn
X= df.iloc[:,:-1].values
Y= df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size= 1/2)

 
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred= regressor.predict(X_test)


plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Price Vs Sqft Living(Training set)')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Price Vs. Sqft Living(Test set)')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.show()
# print(regressor.predict([[1002]]))
a=int(input("What is the house area? "))
print('The Cost for this house is', regressor.predict([[a]]))
