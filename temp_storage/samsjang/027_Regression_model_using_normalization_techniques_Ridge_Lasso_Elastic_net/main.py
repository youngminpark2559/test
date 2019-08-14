# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/027_Regression_model_using_normalization_techniques_Ridge_Lasso_Elastic_net && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# This is personal study note
# Copyright and original reference:
# https://blog.naver.com/samsjang/221004285172

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

# ================================================================================
df=pd.read_csv('../Data/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# ================================================================================
X=df.iloc[:,:-1].values
y=df["MEDV"].values

# ================================================================================
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# ================================================================================
# @ Create regression models

lr=LinearRegression()

# alpha: lambda in fomular
lr=Ridege(alpha=1.0)

lr=Lasso(alpha=1.0)

# ElasticNet()인자 l1_ratio는 Elastic Net을 위한 cost function 식에서 λ2 의 비율을 뜻합니다. 이 값을 1.0으로 설정하면 LASSO와 동일한 효과를 갖습니다.
lr=ElasticNet(alpha=1.0,l1_ratio=0.5)

# ================================================================================
lr.fit(X_train,y_train)

# ================================================================================
y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)

# ================================================================================
plt.scatter(y_train_pred,y_train_pred-y_train,c="blue",marker="o",label="train dataset")
plt.scatter(y_test_pred,y_test_pred-y_test,c="lightblue",marker="s",label="test dataset")

plt.xlabel("pred")
plt.ylabel("residual")
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.legend(loc=2)
plt.show()

# ================================================================================
mse_train=mean_squared_error(y_train,y_train_pred)
mse_test=mean_squared_error(y_test,y_test_pred)
# print("mse_train",mse_train)
# mse_train 19.958219814238046
# print("mse_test",mse_test)
# mse_test 27.19596576688333

# ================================================================================
r2_train=r2_score(y_train,y_train_pred)
r2_test=r2_score(y_test,y_test_pred)
# print("r2_train",r2_train)
# r2_train 0.7645451026942549
# print("r2_test",r2_test)
# r2_test 0.6733825506400181
