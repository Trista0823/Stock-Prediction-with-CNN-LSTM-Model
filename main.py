import numpy as np
import pandas as pd
import tushare as ts  # TuShare is a utility for crawling historical data of China stocks
import time
import math
from pyecharts import Line
from keras.models import Sequential  # before using keras, install Tensorflow first
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
import sklearn.preprocessing as prep


def get_data(category, stock_code, start_date='2006-01-01', end_date=None):
    # category = 'index' or 'stock'
    ts.set_token('e14e782c23aa6584d449f1baaf25ec5ee362fca5a49d9004af087b6e')
    pro = ts.pro_api()
    if category == 'stock':
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)  # 个股
    else:
        df = pro.index_daily(ts_code=stock_code, start_date=start_date, end_date=end_date)  # 指数

    df = df.sort_values(by="trade_date", ascending=True)
    date = df['trade_date']
    del df['ts_code']
    del df['trade_date']
    df = df.reset_index(drop=True)
    col_list = df.columns.tolist()
    col_list.remove('close')
    col_list.append('close')
    df = df[col_list]

    return df, date


def standard_scaler(X_train, X_test, y_train, y_test):
    train_samples, train_nx, train_ny = X_train.shape
    test_samples, test_nx, test_ny = X_test.shape

    X_train = X_train.reshape((train_samples, train_nx * train_ny))
    X_test = X_test.reshape((test_samples, test_nx * test_ny))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    preprocessor_X = prep.StandardScaler().fit(X_train)
    preprocessor_y = prep.StandardScaler().fit(y_train)
    X_train = preprocessor_X.transform(X_train)
    X_test = preprocessor_X.transform(X_test)
    y_train = preprocessor_y.transform(y_train)
    y_test = preprocessor_y.transform(y_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))
    X_test = X_test.reshape((test_samples, test_nx, test_ny))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return X_train, X_test, y_train, y_test, preprocessor_X, preprocessor_y


def preprocess_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.values

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[: int(row), :]
    test = result[int(row):, :]

    X_train = train[:, : -1]
    y_train = train[:, -1][:, -1]

    X_test = test[:, : -1]
    y_test = test[:, -1][:, -1]

    X_train, X_test, y_train, y_test, preprocessor_X, preprocessor_y = standard_scaler(X_train, X_test, y_train, y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test, preprocessor_X, preprocessor_y]


def build_LSTM_model():
    model = Sequential()

    model.add(LSTM(
        units=hidden_layer1,
        return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(
        units=hidden_layer2,
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(
        output_dim=output))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    return model


def build_CNN_LSTM_model():
    model = Sequential()

    model.add(Conv1D(
        filters=hidden_layer1,
        kernel_size=2,
        padding='same'))

    model.add(MaxPooling1D())

    model.add(LSTM(
        units=hidden_layer2,
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(
        units=output))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    return model


def build_CNN_model():
    model = Sequential()

    model.add(Conv1D(
        filters=hidden_layer1,
        kernel_size=2,
        padding='same'))

    model.add(MaxPooling1D())

    model.add(Conv1D(
        filters=hidden_layer2,
        kernel_size=2,
        padding='same'))

    model.add(MaxPooling1D())

    model.add(Flatten())

    model.add(Dense(
        units=output))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    #     print("Compilation Time : ", time.time() - start)
    return model


def build_ANN_model():
    model = Sequential()

    model.add(Dense(
        units=hidden_layer1))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(
        units=hidden_layer2))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(
        units=output))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop" , metrics=['accuracy'])
    return model


def predict_nextday(model, df, window, preprocessor_X, preprocessor_y):
    X_nextday = df[-window:].values
    X_nextday = X_nextday.reshape((1,-1))
    X_nextday = preprocessor_X.transform(X_nextday)
    X_nextday = X_nextday.reshape((window,-1))
    y_nextday = model.predict(X_nextday[np.newaxis,:])
    y_nextday = preprocessor_y.inverse_transform(y_nextday)
    return y_nextday[0][0].round(decimals=2)


def visualize_pyecharts(pred, y_test, date, periods):
    days = date[-periods:]
    days = pd.to_datetime(days,format="%Y/%m/%d")
    pred = pred.flatten()
    pred = pred.astype('float64')
    pred = pred.round(decimals=2)
    line = Line("Test Result")
    line.add("Prediction", days, pred[-periods:], is_smooth=True)
    line.add("Ground Truth", days, y_test[-periods:], is_smooth=True)
    line.render() # generate render.html


if __name__ == '__main__':
    # data sample setting
    category = 'index'
    stock_code = '399300.SZ'
    start_date = '2006-01-01'  

    # Get data
    df, date = get_data(
        category=category,
        stock_code=stock_code,
        start_date=start_date,
        end_date=None)

    if df.shape[0] == 0:
        raise Exception("Invalid stock code!")

    # Model setting
    hidden_layer1 = 10
    hidden_layer2 = 10
    window = 20
    dropout = 0.2
    batch_size = 500
    epochs = 10
    periods = 20  # prediction time period
    output = 1  

    hyperparameter = [hidden_layer1, hidden_layer2, window, batch_size, epochs, periods]
    for i in hyperparameter:
        if not isinstance(i, int) or i <= 0:
            raise Exception("Invalid hyperparameter!")

    if not isinstance(dropout, float) or dropout > 1 or dropout < 0:
        raise Exception("Invalid hyperparameter!")

    # Preprocess
    X_train, y_train, X_test, y_test, preprocessor_X, preprocessor_y = preprocess_data(df, window)

    # Choose model
    model = build_CNN_LSTM_model()

    # Trainning
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0)

    # Predict and Scale back
    pred = model.predict(X_test)
    pred = preprocessor_y.inverse_transform(pred)
    y_test = preprocessor_y.inverse_transform(y_test)

    # Evaluation
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.5f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.1f MSE (%.1f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    diff = []
    ratio = []
    for u in range(len(y_test)):
        pr = pred[u][0]
        ratio.append((pr/y_test[u]) - 1)
        diff.append(abs(y_test[u] - pr))
    diff = np.array(diff)
    ratio = np.array(ratio)
    print('Mean Difference: %.2f' % diff.mean())
    print('Ratio Difference: %.2f' % ratio.mean())

    y_nextday = predict_nextday(model, df, window, preprocessor_X, preprocessor_y)
    print('Prediction Valune on Next day: %.2f' % y_nextday)

    # Visulization
    visualize_pyecharts(pred, y_test, date, periods)
