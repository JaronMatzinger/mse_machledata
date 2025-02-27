from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load(product):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Ayushijain09/Regression-on-COVID-dataset/master/COVID-19_Daily_Testing.csv"
    )
    df.to_csv(product, index=False)


def clean(product, upstream):
    data = pd.read_csv(upstream['load'])
    data['Cases'] = data['Cases'].str.replace(',', '')
    data['Tests'] = data['Tests'].str.replace(',', '')
    data['Cases'] = pd.to_numeric(data['Cases'])
    data['Tests'] = pd.to_numeric(data['Tests'])
    data.to_csv(product, index=False)


def split(product, upstream):
    data = pd.read_csv(upstream['clean'])
    X = data["Tests"].values.reshape(-1, 1)
    y = data["Cases"].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train.to_csv(product['X_train'], index=False)
    X_test.to_csv(product['X_test'], index=False)
    y_train.to_csv(product['y_train'], index=False)
    y_test.to_csv(product['y_test'], index=False)


def linear_regression(product, upstream):
    X_train = pd.read_csv(upstream['split']['X_train'])
    y_train = pd.read_csv(upstream['split']['y_train'])
    reg = LinearRegression().fit(X_train, y_train)
    df_coef = pd.DataFrame(np.transpose(reg.coef_))
    df_coef.to_csv(product, index=False)