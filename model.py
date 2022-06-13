import pickle

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from helpers import *

def train():
    model_obj = models()
    data = get_data()
    X, y = shuffle(data.drop(columns=['SalePrice']).values, data.SalePrice.values, random_state=13)
    X = X.astype(np.float32)
    all_X = X
    all_y = y
    split_datas = train_test_split(
        all_X, all_y, test_size=0.1,random_state=0)
    model_obj.fit(split_datas[0], split_datas[2])
    pickle.dump(model_obj, open("model.pkl", "wb"))
    pickle.dump(split_datas, open("split_datas.pkl", "wb"))
    
    cross_val(X, y)


def predict():
    split_datas = pickle.load(open("split_datas.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    predictions = model.predict(split_datas[1])
    mae, mse, rmse, r2_square = evaluate(split_datas[3], predictions)
    return predictions, mae, mse, rmse, r2_square

