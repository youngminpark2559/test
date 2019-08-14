# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/024_Implement_regression_model_using_least_squares && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# This is personal study note
# Copyright and original reference:
# https://blog.naver.com/samsjang/220994196509

# ================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# ================================================================================
class LinearRegressionGD():
  def __init__(self,eta=0.001,n_iter=20):
    self.eta=eta
    self.n_iter=n_iter
  
  # ================================================================================
  def fit(self,X,y):
    # print("X",X.shape)
    # (506, 1)

    # print("X",X)
    # [[ 4.13671889e-01]
    #  [ 1.94274453e-01]
    #  [ 1.28271368e+00]
    #  [ 1.01630251e+00]

    # 506 data points, one feature (RM: probably number of rooms)

    # ================================================================================
    self.w_=np.zeros(1+X.shape[1])
    # print("self.w_",self.w_.shape)
    # (2,)

    self.cost_=[]

    # ================================================================================
    for i in range(self.n_iter):
      output=self.net_input(X)

      # ================================================================================
      errors=(y-output)
      # print("errors",errors)
      # [ 0.15968566 -0.10152429  1.32424667  1.18275795  1.48750288  0.6712218
      #   0.03996443  0.49708184 -0.65659542 -0.39538548 -0.81985164 -0.39538548
      #  -0.09064054 -0.23212926 -0.47157171 -0.286548    0.06173193 -0.54775795

      # print("errors",errors.shape)
      # (506,)

      # ================================================================================
      # Update weights by using Adaline method
      self.w_[1:]+=self.eta*X.T.dot(errors)
      
      # Update biases by using Adaline method
      self.w_[0]+=self.eta*errors.sum()

      # ================================================================================
      cost=(errors**2).sum()/2.0

      self.cost_.append(cost)
    
    return self
  
  def net_input(self,X):
    weights=self.w_[1:]
    # print("weights",weights)
    # weights [0.]

    # print("weights",weights.shape)
    # weights (1,)

    # ================================================================================
    bias=self.w_[0]
    # print("bias",bias)
    # bias 0.0

    # print("bias",bias.shape)
    # bias ()

    # ================================================================================
    predicted_output_from_simple_regression=np.dot(X,weights)+bias
    # print("predicted_output_from_simple_regression",predicted_output_from_simple_regression)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

    # print("predicted_output_from_simple_regression",predicted_output_from_simple_regression.shape)
    # (506,)

    return predicted_output_from_simple_regression
  
  def predict(self,X):
    return self.net_input(X)

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
# For normalization

sc_x=StandardScaler()
sc_y=StandardScaler()

X_std=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y)

# ================================================================================
y_std=y_std.reshape((-1,))

# ================================================================================
# c lr: create regression model
lr=LinearRegressionGD()

# Optimize regression model
lr.fit(X_std,y_std)

# ================================================================================
# Visualization loss value

# print("lr.n_iter",lr.n_iter)
# 20

x_vals_for_visualization=list(range(1,lr.n_iter+1))
# print("x_vals_for_visualization",x_vals_for_visualization)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

y_vals_for_visualization=lr.cost_
# print("y_vals_for_visualization",y_vals_for_visualization)
# [252.99999999999994, 160.52145703330265, 137.95336332188174, 132.4459360049214, 131.1019254721997, 130.7739385178364, 130.6938978934414, 130.67436509962653, 130.66959839475513, 130.66843514716516, 130.66815127287626, 130.66808199733032, 130.66806509160318, 130.6680609659971, 130.66805995920078, 130.6680597135062, 130.66805965354786, 130.66805963891588, 130.66805963534517, 130.66805963447376]

plt.plot(x_vals_for_visualization,y_vals_for_visualization)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()

# ================================================================================
def lin_regplot(X,y,model):
  plt.scatter(X,y,c='b')
  plt.plot(X,model.predict(X),c='r')

lin_regplot(X_std,y_std,lr)
plt.xlabel("Avg of RM (normalized)")
plt.ylabel("House price, 1000$ unit (normalized)")
plt.show()

# ================================================================================
num_rooms=5.0

# Normalize X
num_rooms_std=sc_x.transform([[num_rooms]])
# print("num_rooms_std",num_rooms_std)
# [[-1.83016553]]

# ================================================================================
house_val_std=lr.predict(num_rooms_std)

# inverse normalze y
house_val=sc_y.inverse_transform(house_val_std)

# ================================================================================
print("num_rooms",num_rooms)
print("house_val",house_val*1000)
# num_rooms 5.0
# house_val [10839.93288858]
