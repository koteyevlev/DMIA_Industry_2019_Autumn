#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'koteyevlev@mail.com'
# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 1}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau

    def loss(self, curr_pred, y_data):
        return (curr_pred - y_data) ** 2
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            # grad = 0. # TODO.
            grad = - 2 * (curr_pred - y_data)
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, grad) # TODO
            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            #curr_pred += self.tau * algo.predict(X_data)# TODO
            #print(np.sum(self.loss(curr_pred, y_data)))
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
       #print(res)
        return res > 0.




from sklearn.model_selection import cross_val_score
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import imp
import signal
import pandas
import traceback
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
data_path=SCRIPT_DIR + '/HR.csv'
df = pandas.read_csv(data_path)
target = np.array(df['left'])
data = np.array(df[[c for c in df if c != 'left']])

algo = SimpleGB(tree_params_dict=TREE_PARAMS_DICT,iters=100,tau=TAU)
# print(np.mean(cross_val_score(algo, data, target, cv=3, scoring='accuracy')))
lr = GradientBoostingClassifier()
lr.fit(data, target)
algo.fit(data, target)
print("orig - ", accuracy_score(lr.predict(data), target))
print("my accuracy - ", accuracy_score(algo.predict(data), target))