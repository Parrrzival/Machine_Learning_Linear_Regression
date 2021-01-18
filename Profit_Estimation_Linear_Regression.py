import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

companies = pd.read_csv(r'C:\Users\LENOVO\Downloads\profit_estimation_of_companies-master\profit_estimation_of_companies-master\1000_Companies.csv')
print(companies.head())


X = companies.iloc[:,:-1].values
y = companies.iloc[:,4].values

print("Map = ", sns.heatmap(companies.corr()))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("Coef = ",regressor.coef_,"Intercept = ",regressor.intercept_)

from sklearn.metrics import r2_score
print("R2 Score = ",r2_score(y_test,y_pred))