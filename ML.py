import numpy as np
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler # 执行特征缩放


df_train = pd.read_excel(io='../for-non-linear.xlsx',sheet_name='Xcal')
df_test = pd.read_excel(io='../for-non-linear.xlsx',sheet_name='Xtest')
np_train_data = np.array(df_train)[:,3:]
np_train_label = np.array(df_train)[:,:3]
np_test_data = np.array(df_test)[:,3:]
np_test_label = np.array(df_test)[:,:3]

stdx = StandardSca = StandardScaler()
np_train_d = stdx.fit_transform(np_train_data)
np_test_d = stdx.fit_transform(np_test_data)
np_train_l = stdx.fit_transform(np_train_label)
np_test_l = stdx.fit_transform(np_test_label)

def try_different_method(model, stdx=None):
    if stdx:
        model.fit(np_train_d,np_train_l)
        result = model.predict(np_test_d)
        print("r2: ", r2_score(np_test_l[:,0], result[:,0]))
        print("r2: ", r2_score(np_test_l[:,1], result[:,1]))
        print("r2: ", r2_score(np_test_l[:,2], result[:,2]))
    else:
        model.fit(np_train_data,np_train_label)
        result = model.predict(np_test_data)
        print("r2: ", r2_score(np_test_label[:,0], result[:,0]))
        print("r2: ", r2_score(np_test_label[:,1], result[:,1]))
        print("r2: ", r2_score(np_test_label[:,2], result[:,2]))

def try_different_method_multi(model, stdx=None):
    if stdx:
        model.fit(np_train_d,np_train_l[:,0])
        result = model.predict(np_test_d)
        print("r2: ", r2_score(np_test_l[:,0], result))
        model.fit(np_train_d,np_train_l[:,1])
        result = model.predict(np_test_d)
        print("r2: ", r2_score(np_test_l[:,1], result))
        model.fit(np_train_d,np_train_l[:,2])
        result = model.predict(np_test_d)
        print("r2: ", r2_score(np_test_l[:,2], result))
    else:
        model.fit(np_train_data,np_train_label[:,0])
        result = model.predict(np_test_data)
        print("r2: ", r2_score(np_test_label[:,0], result))
        model.fit(np_train_data,np_train_label[:,1])
        result = model.predict(np_test_data)
        print("r2: ", r2_score(np_test_label[:,1], result))
        model.fit(np_train_data,np_train_label[:,2])
        result = model.predict(np_test_data)
        print("r2: ", r2_score(np_test_label[:,2], result))


print("#####decision tree####")
from sklearn import tree
model = tree.DecisionTreeRegressor()
try_different_method(model)
print("####linear####")
from sklearn import linear_model
model = linear_model.LinearRegression()
try_different_method(model)
print("###SVM####")
from sklearn import svm
model = svm.SVR(kernel='rbf', C=100)
try_different_method_multi(model, 1)
print("####KNN###")
from sklearn import neighbors
model = neighbors.KNeighborsRegressor(n_neighbors = 2)
try_different_method(model, 1)
print("####RF####")
from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=100)#这里使用20个决策树
try_different_method(model)
print("####Adaboost####")
from sklearn import ensemble
model = ensemble.AdaBoostRegressor(n_estimators=300)#这里使用50个决策树
try_different_method_multi(model)
print("####GBRT####")
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(n_estimators=400)#这里使用100个决策树
try_different_method_multi(model)
print("####Bagging####")
from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor()
try_different_method(model)
print("####xtraTree####")
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor()
try_different_method(model)
print("####PLS####")
from sklearn.cross_decomposition import PLSRegression
model = PLSRegression(n_components = 16)
try_different_method(model)