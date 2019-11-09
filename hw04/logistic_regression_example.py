# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'koteyevlev@mail.com'

LR_PARAMS_DICT = {
    'C': 10.,
    'random_state': 777,
    'iters': 1000,
    'batch_size': 1000,
    'step': 0.01
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X @ theta)
        epsilon = 1e-5
        cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        #print(cost,len(h))
        return cost

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    # производная регуляризатора
    def der_reg(self):
        # TODO
        #print(2 * self.C * self.w)
        return 2 * self.w / self.C

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # s.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        # TODO
        ders_w = 0.
        der_w0 = 0.

        ders_w = (np.dot(x.T, self.sigmoid(self.__predict(x)) - y) / self.batch_size)
        der_w0 = ((np.dot(np.ones(len(x)), self.sigmoid(self.__predict(x)) - y) / self.batch_size))
        #der_w0 = self.w0 - 0.00109561
        #print((self.__predict(x) - y).shape())
        #print(der_w0)
        #print()
        # для масштаба возвращаем средний градиент по пачке
        # TODO
        return ders_w, der_w0

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step
        #print(self.compute_cost(X_train, y_train, self.w))

        # метод fit для sklearn должен возвращать self
        #print(self.w)
        return self




from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import warnings
warnings.filterwarnings("ignore")

model = MyLogisticRegression(**LR_PARAMS_DICT)
X_data, y_data = make_classification(
            n_samples=10000, n_features=20, 
            n_classes=2, n_informative=20, 
            n_redundant=0,
            random_state=75
        )
#print(np.mean(cross_val_score(model, X_data, y_data, cv=2, scoring='accuracy')))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    print(cost,len(h))
    return cost


model.fit(X_data, y_data)
lr = LogisticRegression(C=10., random_state=777, max_iter=1000)
lr.fit(X_data, y_data)
#print(len(X_data[0]), len(lr.coef_[0]))
#print("orig - ", accuracy_score(lr.predict(X_data), y_data))
#print("my accuracy - ", accuracy_score(model.predict(X_data), y_data))
