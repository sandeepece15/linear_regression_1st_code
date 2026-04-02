import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\machine learning\LinearRegressionHW\Food_Delivery_Times.csv')


col=['Weather','Traffic_Level','Time_of_Day']

for c in col:
    df[c]=df[c].fillna(df[c].mode()[0])


df['Courier_Experience_yrs']=df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())
# print(df.isnull().sum())

# print(df.head())

df=pd.get_dummies(df,columns=['Vehicle_Type','Weather','Traffic_Level','Time_of_Day'],drop_first=True)

x=df.drop('Delivery_Time_min',axis=1)
y=df['Delivery_Time_min']


# train test split --to split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# train linear regression model
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)


# to make prediction 
y_pred=model.predict(x_test)

# evaluate the model

from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("MSF: ",mse) # error 
print("R2 score :",r2) # accuracy


from sklearn.linear_model import LinearRegression, Ridge, Lasso

ridge=Ridge()
ridge.fit(x_train,y_train)
pred1=ridge.predict(x_test)
r21=r2_score(y_test,pred1)

lasso=Lasso()
lasso.fit(x_train,y_train)
pred2=lasso.predict(x_test)
r22=r2_score(y_test,pred2)
