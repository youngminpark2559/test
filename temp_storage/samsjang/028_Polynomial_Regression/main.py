# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/028_Polynomial_Regression && \
# rm e.l && python main.py \
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
X=np.array([258.0,270.0,294.0,320.0,342.0,368.0,396.0,446.0,480.0,586.0])[:,np.newaxis]
y=np.array([236.4,234.4,252.8,298.6,314.2,342.2,360.8,368.0,391.2,390.8])

# ================================================================================
lr=LinearRegression()
pr=LinearRegression()

# ================================================================================
# x^0,x^1,x^2
quadratic=PolynomialFeatures(degree=2)
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

# ================================================================================
# print("X",X.shape)
# X (10, 1)
# print("y",y.shape)
# y (10,)

# @ Train linear regression model
lr.fit(X,y)

# ================================================================================
# @ Generate data to make "linear regression model's straight line"
X_fit=np.arange(250,600,10)[:,np.newaxis]
# print("X_fit",X_fit)
# [[250]
#  [260]
#  [270]

# print("X_fit",X_fit.shape)
# (35, 1)

# @ Make prediction from data
y_lin_fit=lr.predict(X_fit)
# print("y_lin_fit",y_lin_fit)
# [250.86164718 256.26469105 261.66773493 267.0707788  272.47382268
#  277.87686655 283.27991043 288.6829543  294.08599818 299.48904205

# print("y_lin_fit",y_lin_fit.shape)
# (35,)

# ================================================================================
# print("X_quad",X_quad.shape)
# X_quad (10, 3)

# print("y",y.shape)
# y (10,)

# @ Train regression model (polynomial)
pr.fit(X_quad,y)

# ================================================================================
# @ Generate data to make "polynomial regression model's curve line"
X_fit_quad=quadratic.fit_transform(X_fit)
# print("X_fit_quad",X_fit_quad)
# [[1.000e+00 2.500e+02 6.250e+04]
#  [1.000e+00 2.600e+02 6.760e+04]
#  [1.000e+00 2.700e+02 7.290e+04]

# print("X_fit_quad",X_fit_quad.shape)
# X_fit_quad (35, 3)

y_quad_fit=pr.predict(X_fit_quad)
# print("y_quad_fit",y_quad_fit)
# [215.86619864 228.37947485 240.44271083 252.0559066  263.21906215
#  273.93217748 284.19525259 294.00828748 303.37128216 312.28423661

# print("y_quad_fit",y_quad_fit.shape)
# y_quad_fit (35,)

# ================================================================================
y_linear_pred=lr.predict(X)

y_quad_pred=pr.predict(X_quad)

# ================================================================================
mse_lin=mean_squared_error(y,y_linear_pred)

mse_quad=mean_squared_error(y,y_quad_pred)

# ================================================================================
r2_lin=r2_score(y,y_linear_pred)

r2_quad=r2_score(y,y_quad_pred)

# ================================================================================
print("mse_lin",mse_lin)
print("mse_quad",mse_quad)

print("r2_lin",r2_lin)
print("r2_quad",r2_quad)

# ================================================================================
# Draw data
plt.scatter(X,y,label="train data")

# @ Draw straight line
plt.plot(X_fit,y_lin_fit,label="linear fit",linestyle="--")

# @ Draw curve line
plt.plot(X_fit,y_quad_fit,label="quadratic fit")

plt.legend(loc=2)

plt.show()

