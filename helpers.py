import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

def get_data():
    data = pd.read_csv(r"house.txt")

    dropNanByDensity = lambda dataFrame, density: [ key for key in dataFrame if (dataFrame[key].isna().sum() / dataFrame.shape[0]) > density ] 
    getDummyColumns = lambda dataFrame : [ key for key in dataFrame if data[key].dtype == 'O' ] 
    getNumericColumns = lambda dataFrame : [ key for key in dataFrame if data[key].dtype != 'O' ] 


    a = data[dropNanByDensity(data,0.005)].dropna() 
    data = data.drop(a, axis=1)

    clf = IsolationForest(  max_samples="auto", random_state = 1, contamination= 'auto')
    preds = clf.fit_predict(data[getNumericColumns(data)].dropna())
    data = data.drop(labels=np.where(preds == -1)[0], axis=0)
    data = data.interpolate(method='spline', order=2)



    data = pd.get_dummies(data, prefix = getDummyColumns(data))
    return data

def scatter(df):
    [df.plot(x = key, y = "SalePrice", kind = "scatter") for key in df.keys()]
   
def models():
    def LGBM():
        model=lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.035, n_estimators=2177, max_bin=50, bagging_fraction=0.65,bagging_freq=5, bagging_seed=7, 
                                    feature_fraction=0.201, feature_fraction_seed=7,n_jobs=-1)
        return model

    def ExtraTreesRegressor():
        model = ExtraTreesRegressor(n_estimators=100, random_state=0)
        return model

    def XGB():
        model = XGBRegressor()
        return model

    def GBR():
        model = GradientBoostingRegressor(n_estimators=1992, learning_rate=0.03005, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=14, loss='huber', random_state =42)
        return model

    def RandomForest():
        model = RandomForestRegressor(n_estimators=1000)
        return model


    def LinearReg():
        model = linear_model.LinearRegression()
        return model


    def SVM():
        model = SVR(kernel='rbf', C=1000000, epsilon=0.001)
        return model

    def Ridge():
        model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
        return model

    def Lasso():
        model = Lasso(alpha=0.1, precompute=True, positive=True, selection='random',random_state=42)
        return model

    def ElasticNet():
        model =  ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
        return model

    model = LGBM()
    return model

def cross_val(X, y):
    model = models()
    scores = cross_val_score(model, X, y, cv=10) #all 
    print("Accuracy: ",scores.mean())
    
def evaluate(actual, predicted):
    mae = metrics.mean_absolute_error(actual, predicted)
    mse = metrics.mean_squared_error(actual, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(actual, predicted))
    r2_square = metrics.r2_score(actual, predicted)
    return mae, mse, rmse, r2_square

