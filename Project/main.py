__author__ = 'Sandra'

import pandas
from sklearn import preprocessing
import numpy
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

numpy.set_printoptions(threshold=numpy.nan)


def shuffle_and_split_data():
    data = pandas.read_csv('dataset.csv')
    ts = data.shape
    df = pandas.DataFrame(data)
    shuffled = df.reindex(numpy.random.permutation(df.index))
    percent = int((ts[0] / 100.0) * 90)
    shuffled[percent:].to_csv('test.csv')
    shuffled[:percent].to_csv('train.csv')

def read_data(file_name):
    data = pandas.read_csv(file_name)

    names = data['name'].tolist()
    channels = data['channel'].tolist()
    genres = data['genre'].tolist()
    seasons = data['season'].tolist()
    episodes = data['episode'].tolist()
    years = data['year'].tolist()
    days = data['day'].tolist()
    time = data['time'].tolist()
    minutes = data['minutes'].tolist()
    viewers = data['viewers'].tolist()
    ratings = data['rating'].tolist()

    encoder = preprocessing.LabelEncoder()
    encoder.fit(names)
    names_labeled = encoder.transform(names)

    encoder.fit(channels)
    channels_labeled = encoder.transform(channels)

    encoder.fit(genres)
    genres_labeled = encoder.transform(genres)

    encoder.fit(years)
    years_labeled = encoder.transform(years)

    encoder.fit(days)
    days_labeled = encoder.transform(days)

    encoder.fit(time)
    time_labeled = encoder.transform(time)

    x = numpy.column_stack([names_labeled, channels_labeled, genres_labeled, episodes, years_labeled, days_labeled, time_labeled, minutes, viewers])
    y = ratings

    return x, y


def ensemble(x_train, y_train, x_test, y_test):
    #en = RandomForestRegressor(n_estimators=100, max_depth=None, max_features=9, min_samples_split=2, random_state=13)
    #en = DecisionTreeRegressor(criterion='mae', splitter='best', max_depth=None, max_features=9)
    en = ExtraTreesRegressor(n_estimators=700, criterion='mse', max_depth=None, min_samples_split=2, max_features=9, random_state=6)
    en.fit(x_train, y_train)
    y_pred = en.predict(x_test)

    rmse = get_rmse(y_test, y_pred)


    return rmse

def normalize_data(train):
    train = preprocessing.normalize(train, norm='l1')
    return train


def get_rmse(y_true, y_predict):
    sum = 0.0
    for i in range(0, len(y_true)):
        #print(str(y_true[i]) + " | " + str(y_predict[i]))
        sum += math.pow((y_true[i] - y_predict[i]), 2)
    rmse = math.sqrt(float(sum)/float(len(y_true)))
    return rmse



if __name__ == '__main__':
    x_train, y_train = read_data('train.csv')
    x_test, y_test = read_data('test.csv')

    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    rmse = ensemble(x_train, y_train, x_test, y_test)
    print(rmse)