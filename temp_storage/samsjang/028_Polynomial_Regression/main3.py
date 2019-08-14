# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/028_Polynomial_Regression && \
# rm e.l && python main3.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# This is personal study note
# Copyright and original reference:
# https://blog.naver.com/samsjang/221006905415

# ================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# ================================================================================
df=pd.read_csv('../Data/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# ================================================================================
X=df[["LSTAT"]].values
y=df["MEDV"].values

# ================================================================================
lr=LinearRegression()

# ================================================================================
# @ Preprocess data

X_log=np.log(X)
y_sqrt=np.log(y)

# ================================================================================
# @ Create line for visualization

X_fit=np.arange(X_log.min()-1,X_log.max()+1,1)[:,np.newaxis]

# ================================================================================
# @ Train regression model

lr.fit(X_log,y_sqrt)

# @ Make prediction
y_lin_fit=lr.predict(X_fit)

# @ Calculate r^2 score
l_r2=r2_score(y_sqrt,lr.predict(X_log))
# print("l_r2",l_r2)
# 0.6772632131699524

# ================================================================================
# @ Draw visualization

# @ Draw data
plt.scatter(X_log,y_sqrt,c='lightgray',label="train data")

# @ Draw straight line
plt.plot(X_fit,y_lin_fit,c="blue",lw=3,label="linear fit, degree=1, $R^2=%.2f$"%l_r2)

plt.xlabel("log(LSTAT_value)")
plt.ylabel("$\sqrt{MEDV}$")
plt.legend()
plt.show()
