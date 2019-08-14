# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/028_Polynomial_Regression && \
# rm e.l && python main2.py \
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
# x^0,x^1,x^2
quadratic=PolynomialFeatures(degree=2)

# x^0,x^1,x^2,x^3
cubic=PolynomialFeatures(degree=3)

# ================================================================================
X_quad=quadratic.fit_transform(X)
# print("X_quad",X_quad)
# [[1.00000e+00 2.58000e+02 6.65640e+04]
#  [1.00000e+00 2.70000e+02 7.29000e+04]
#  [1.00000e+00 2.94000e+02 8.64360e+04]
#  [1.00000e+00 3.20000e+02 1.02400e+05]
#  [1.00000e+00 3.42000e+02 1.16964e+05]
#  [1.00000e+00 3.68000e+02 1.35424e+05]
#  [1.00000e+00 3.96000e+02 1.56816e+05]
#  [1.00000e+00 4.46000e+02 1.98916e+05]
#  [1.00000e+00 4.80000e+02 2.30400e+05]
#  [1.00000e+00 5.86000e+02 3.43396e+05]]

# print("X_quad",X_quad.shape)
# (10, 3)

X_cubic=cubic.fit_transform(X)
# print("X_cubic",X_cubic)
# [[  1.         4.98      24.8004   123.505992]
#  [  1.         9.14      83.5396   763.551944]
#  [  1.         4.03      16.2409    65.450827]

# print("X_cubic",X_cubic.shape)
# (506, 4)

# ================================================================================
# Perform "1 degree regression"

X_fit=np.arange(X.min(),X.max(),1)[:,np.newaxis]
# print("X_fit",X_fit)
# [[ 1.73]
#  [ 2.73]
#  [ 3.73]

# print("X_fit",X_fit.shape)
# X_fit (37, 1)

# @ Train
lr.fit(X,y)

# @ For visualized line
y_lin_fit=lr.predict(X_fit)

# @ Make prediction
y_linear_pred=lr.predict(X)

# @ Coefficient of determinant by r^2
l_r2=r2_score(y,y_linear_pred)
# print("l_r2",l_r2)
# 0.5441462975864799

# ================================================================================
# @ Perform "2 degree polynomial regression"

# @ Train
lr.fit(X_quad,y)

# @ For curve line in visualization
y_quad_fit=lr.predict(quadratic.fit_transform(X_fit))

# @ r2 score
q_r2=r2_score(y,lr.predict(X_quad))

# ================================================================================
# Perform "3 degree polynomial regression"

# @ Train
lr.fit(X_cubic,y)

# @ For curve line in visualization
y_cubic_fit=lr.predict(cubic.fit_transform(X_fit))

# r2 score
c_r2=r2_score(y,lr.predict(X_cubic))
# print("c_r2",c_r2)
# 0.657847640589572

# ================================================================================
# @ Draw visualization

# @ Draw data
plt.scatter(X,y,label="train data",c="lightgray")

# @ Draw straight line
plt.plot(X_fit,y_lin_fit,label="linear fit, degree=1, $R^2=%.2f$"%l_r2,linestyle=":",c="blue",lw=3)

# @ Draw 2 degree curve line
plt.plot(X_fit,y_quad_fit,label="quad fit, degree=2, $R^2=%.2f$"%q_r2,linestyle=":",c="red",lw=3)

# @ Draw 3 degree curve line
plt.plot(X_fit,y_cubic_fit,label="cubic fit, degree=3, $R^2=%.2f$"%c_r2,linestyle="--",c="green",lw=3)

plt.xlabel("Percentage of population")
plt.ylabel("House price (1000$ unit)")

plt.legend(loc=1)

plt.show()
