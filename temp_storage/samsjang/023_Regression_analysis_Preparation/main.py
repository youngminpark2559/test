# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/samsjang/023_Regression_analysis_Preparation && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================================================================
df=pd.read_csv('./Data/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
        
# print("df.tail()",df.tail())
#         CRIM   ZN  INDUS  CHAS    NOX  ...     TAX  PTRATIO       B  LSTAT  MEDV
# 501  0.06263  0.0  11.93     0  0.573  ...   273.0     21.0  391.99   9.67  22.4
# 502  0.04527  0.0  11.93     0  0.573  ...   273.0     21.0  396.90   9.08  20.6
# 503  0.06076  0.0  11.93     0  0.573  ...   273.0     21.0  396.90   5.64  23.9
# 504  0.10959  0.0  11.93     0  0.573  ...   273.0     21.0  393.45   6.48  22.0
# 505  0.04741  0.0  11.93     0  0.573  ...   273.0     21.0  396.90   7.88  11.9

# [5 rows x 14 columns]

# ================================================================================
# @ Draw pairplot (correlation visualization)

cols=["LSTAT","INDUS","NOX","RM","MEDV"]
sns.pairplot(df[cols],size=2.5)
plt.show()

# Visualzied correlations between features

# ================================================================================
# @ Calculate correlation coefficient

cm=np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.show()
