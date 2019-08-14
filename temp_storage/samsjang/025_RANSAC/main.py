# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/025_RANSAC && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# This is personal study note
# Copyright and original reference:
# https://blog.naver.com/samsjang/221003286930

# ================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

# ================================================================================
df=pd.read_csv('../Data/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# ================================================================================
# c X: one feature
X=df[["RM"]].values

# c X: one label
y=df[["MEDV"]].values

y=y.reshape((-1,1))

# ================================================================================
# Create RANSAC

ransac=RANSACRegressor(
  LinearRegression(),max_trials=100,min_samples=50,
  # residual_metric=lambda x:np.sum(np.abs(x),axis=1),residual_threshold=5.0,random_state=0)
  residual_threshold=5.0,random_state=0)

# Train ransac
ransac.fit(X,y)

# ================================================================================
# Use trained ransac

inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)

line_X=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_X[:,np.newaxis])

# ================================================================================
plt.scatter(X[inlier_mask],y[inlier_mask],c='blue',marker='o',label='inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],c='lightgreen',marker='s',label='outliers')

plt.plot(line_X,line_y_ransac,c='red')
plt.xlabel("Avg of RM (normalized)")
plt.ylabel("House price, 1000$ unit (normalized)")
plt.title("1978 year, Boston house prices")
plt.legend(loc=2)
plt.show()

# ================================================================================
slope=ransac.estimator_.coef_[0]
print("slope",slope)

intercept=ransac.estimator_.intercept_
print("intercept",intercept)
