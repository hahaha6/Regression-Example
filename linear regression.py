# import numpy as np
# from sklearn.linear_model import LinearRegression
# path = 'lineardata/Pdata12_1.txt'
# a = np.loadtxt(path)
# md = LinearRegression().fit(a[:,:2],a[:,2])
# y = md.predict(a[:,:2])
# b0 = md.intercept_;b12 = md.coef_
# R2 = md.score(a[:,:2],a[:,2])
# print("b0=%.4f\nb12=%.4f%10.4f"%(b0,b12[0],b12[1]))
# print("拟合优度R^w=%.4f"%R2)
# import numpy as np
# import statsmodels.api as sm
# path = 'lineardata/Pdata12_1.txt'
# a = np.loadtxt(path)
# d = {'x1':a[:,0],'x2':a[:,1],'y':a[:,2]}
# md = sm.formula.ols('y~x1+x2',d).fit()
# print(md.summary(),'\n-----------\n')
# ypred = md.predict({'x1':a[:,0],'x2':a[:,1]})
# print(ypred)
# import numpy as np
# import statsmodels.api as sm
# path = 'lineardata/Pdata12_3.txt'
# a = np.loadtxt(path)
# x = a[:,:3]
# X = sm.add_constant(x)
# md = sm.OLS(a[:,3],X).fit()
# b = md.params
# y = md.predict(X)
# print(md.summary())
# print("相关系数矩阵:",np.corrcoef(x.T))
# X1 = sm.add_constant(a[:,0])
# md1 = sm.OLS(a[:,2],X1).fit()
# print("回归系数为:",md1.params)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge , RidgeCV
# from scipy.stats import zscore
# path = 'lineardata/Pdata12_3.txt'
# a = np.loadtxt(path)
# n = a.shape[1]-1
# aa = zscore(a)
# x = aa[:,:n]
# y = aa[:,n]
# b = []
# kk = np.logspace(-4,0,100)
# for k in kk:
#     md = Ridge(alpha = k).fit(x,y)
#     b.append(md.coef_)
# st = ['s-r','*-k','p-b']
# for i in range(3):plt.plot(kk, np.array(b)[:,i],st[i])
# plt.legend(['x_1','x_2','x_3'],fontsize=15);plt.show()
# mdcv = RidgeCV(alphas = np.logspace(-4,0,100)).fit(x,y)
# print("最优的alpha = ",mdcv.alpha_)
# md0 = Ridge(0.4).fit(x,y)
# cs0 = md0.coef_
# print("标准化数据的所有回归系数为:",cs0)
# mu = np.mean(a, axis = 0)
# s = np.std(a,axis = 0,ddof=1)
# params = [mu[-1]-s[-1]*sum(cs0*mu[:-1]/s[:-1]),s[-1]*cs0/s[:-1]]
# print("元数据的回归系数为:",params)
# print("拟合优度",md0.score(x,y))

# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import logspace
# from sklearn.linear_model import Lasso, LassoCV
# from scipy.stats import zscore
# plt.rc('font', size = 16)
# path = 'lineardata/Pdata12_3.txt'
# a = np.loadtxt(path)
# n = a.shape[1]-1
# aa = zscore(a)
# x = aa[:,:n]
# y = aa[:,n]
# b = []
# kk = logspace(-4,0,100)
# for k in kk:
#     md = Lasso(alpha = k).fit(x,y)
#     b.append(md.coef_)
# print(np.array(b).shape)
# st = ['s-r','*-k','p-b']
# for i in range(3):plt.plot(kk,np.array(b)[:,i],st[i]);
# plt.legend(['x_1','x_2','x_3'],fontsize = 15);plt.show()
# mdcv = LassoCV(alphas = np.logspace(-4,0,100)).fit(x,y)
# print("最优的alpha=",mdcv.alpha_)
# md0 = Lasso(0.21).fit(x,y)
# cs0 = md0.coef_
# print("标准化数据所有回归系数为:",cs0)
# mu = np.mean(a,axis=0);s = np.std(a,axis = 0,ddof = 1)

# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from sklearn.linear_model import Lasso
# from scipy.stats import zscore
# path = 'lineardata/Pdata12_6.txt'
# a = np.loadtxt(path)
# n = a.shape[1]-1
# print(n)
# x = a[:,:n]
# X = sm.add_constant(x)
# md = sm.OLS(a[:,n],X).fit()
# print(md.summary())
# aa = zscore(a)
# x = aa[:,:n];y = aa[:,n]
# b = []
# kk = np.logspace(-4,0,100)
# for k in kk :
#     md = Lasso(alpha = k).fit(x, y)
#     b.append(md.coef_)
# st = ['s-r','*-k','p-b','^-y']
# for i in range(n):plt.plot(kk,np.array(b)[:,i],st[i])
# plt.legend(['x_1','x_2','x_3','x_4'],fontsize = 15)
# plt.show()

#logistic
# import numpy as np
# import statsmodels.api as sm
# path = 'lineardata/Pdata12_7_1.txt'
# a = np.loadtxt(path)
# x = a[:,0]; pi = a[:,2]/a[:,1]
# X = sm.add_constant(x); yi = np.log([pi/(1-pi)])
# md = sm.OLS(yi,X).fit()
# print(md.summary())
# b = md.params
# p0 = 1/(1+np.exp(-np.dot(b,[1,9])))
# print("所求的概率P0=%.4f"%p0)
import numpy as np
import statsmodels.api as sm
path = 'lineardata/Pdata12_9.txt'
a = np.loadtxt(path)
n = a.shape[1]
x = a[:,:n-1];y = a[:,n-1]
md = sm.Logit(y,x)
ms = md.fit(method = "bfgs")
print(ms.params,'\n--------\n')
print(ms.summary2())
print(ms.predict([[-49.2, -17, 0.3],[40.6, 26.4, 1.8]]))


