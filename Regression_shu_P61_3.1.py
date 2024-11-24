# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:37:36 2024

@author: a5261

房总回归分析作业P61，3
"""
import numpy as np
import statsmodels.api as sm
#import scipy.stats as stats
#from statsmodels.sandbox.regression.predstd import wls_prediction_std

col1 = np.array([70, 75, 65, 74, 72, 68, 78, 66, 70, 65])
col2 = np.array([35, 40, 40, 42, 38, 45, 42, 36, 44, 42])
col3 = np.array([1, 2.4, 2, 4, 1.2, 1.5, 4, 2, 4.2, 4])

p = 3
n = 10
X = np.array([col1,col2,col3]).T
Y = np.array([160, 260, 210, 265, 240, 220, 275, 160, 275, 250])

X = sm.add_constant(X)   # 添加截距系数列
model = sm.OLS(Y,X).fit()


# （1）（3）（4） 回归系数，建立模型，检验
print("（1）模型参数:\n",model.summary())
# （2）
model.mse_resid   # 均方误差（MSE） = 方差估计值。残差平方和SSE，残差标准误/残差标准差/Residual Std. Error=sqrt(MSE)
print(f"\n（2）方差的估计为{model.mse_resid}")


# 调整模型 删除变量x3
model2 = sm.OLS(Y,X[:,:3]).fit()
print("\n \n \n\n \n \n（5）调整后的模型参数：\n",model2.summary())




# prediction result
pred_result = model2.get_prediction(X[:,:3])
# y_pred = model2.predict(X[:,:3])   # point estimation values
y_pred,y_interval = pred_result.predicted_mean, pred_result.conf_int()
print(f"\n \n \n\n \n \nY的点估计值为\n{y_pred}\n\n Y的95%预测区间为\n{y_interval}")




"""
# 获取预测值和预测值标准误差
resid = model.resid   # 残差
y_pred3, y_std, cov_matrix =wls_prediction_std(model2, exog = X[:, :3])   # 加权最小二乘法（WLS）
df = n-2-1
t_value = stats.t.ppf((1 + 0.90) / 2,df)   # t(df) 5%上分位数

y_interval_90 = np.array([y_pred - t_value*y_std,y_pred + t_value*y_std]).T
"""



#(7)
X0 = np.array([90,57,6.2])
X0 = np.insert(X0, 0, 1).reshape(1, -1)   # 添加截距项   -1自动计算列数
y0 = model2.predict(X0[:,:3])
print("\n(7)预测值为：", y0)


# 均值的置信区间


