# Import of all the classifiers and regressors in sklearn 
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

from sklearn.isotonic import IsotonicRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

# XGBoost library
from xgboost import XGBRegressor
from xgboost import XGBClassifier

# scikit-elm library
from skelm import ELMRegressor
from skelm import ELMClassifier

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, balanced_accuracy_score

import numpy as np

DEFAULT_REGRESSION_METRICS = [
    "R2", "MSE", "MAE"
]

DEFAULT_CLASSIFICATION_METRICS = [
    "ACC", "F1W"
]

DEFAULT_REGRESSION_MODELS = [
    ("Dummy Mean", DummyRegressor(strategy="mean"), {}),

    ("Dummy Median", DummyRegressor(strategy="median"), {}),

    ("Linear Regression", LinearRegression(), {}),

    ("Decision Tree", DecisionTreeRegressor(), 
        {
            "criterion": ["friedman_mse", "squared_error", "absolute_error", "poisson"],
            "max_depth": np.linspace(1, 10, 10).astype(int)
        }),

    ("SVR", SVR(), {}),

    ("Linear SVR", LinearSVR(dual="auto"), {}),

    ("Ridge", Ridge(), 
        {
            "alpha": np.geomspace(1e-7, 10, 10)
        }),

    ("Passive Aggressive", PassiveAggressiveRegressor(), {}),

    ("KNN", KNeighborsRegressor(), {}),

    ("MLP", MLPRegressor(), {}),

    ("Gaussian Process", GaussianProcessRegressor(), {}),

    ("Random Forest", RandomForestRegressor(), {}),

    ("AdaBoost", AdaBoostRegressor(), {}),

    ("Gradient Boosting", GradientBoostingRegressor(), {}),

    ("Ridge CV", RidgeCV(), {}),

    ("Lasso CV", LassoCV(), {}),

    ("Elastic Net CV", ElasticNetCV(), {}),

]

DEFAULT_CLASSIFICATION_MODELS = [
    ("Dummy", DummyClassifier(strategy="most_frequent"), {}),

    ("Decision Tree", DecisionTreeClassifier(random_state=0), 
        {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": np.linspace(1, 10, 10).astype(int)
        }),

    ('SVC', SVC(gamma='auto', kernel='rbf'), {}),

    ('Linear SVC', LinearSVC(random_state=0, tol=1e-5, dual="auto"), {}),

    ('Perceptron', Perceptron(random_state=0), {}),

    ('Logistic Regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto'), {}),

    ('Ridge Classifier', RidgeClassifier(random_state=0), {}),

    ('SGD Classifier', SGDClassifier(random_state=0), {}),

    ('Passive Aggressive Classifier', PassiveAggressiveClassifier(random_state=0), {}),
    
    ('K Neighbors Classifier', KNeighborsClassifier(), {}),

    ('MLP Classifier', MLPClassifier(random_state=0),
        {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)], 
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'], 
            'alpha': [0.0001, 0.05], 
            'learning_rate': ['constant', 'adaptive'], 
        }),

    ('Gaussian Process Classifier', GaussianProcessClassifier(random_state=0), {}),

    ('Random Forest Classifier', RandomForestClassifier(random_state=0), {}),

    ('AdaBoost Classifier', AdaBoostClassifier(random_state=0), {}),

    ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=0), {}),

    ('Extra Trees Classifier', ExtraTreesClassifier(random_state=0), {}),

    ('Bagging Classifier', BaggingClassifier(random_state=0), {}),

    ("Decision Tree CV", DecisionTreeClassifier(random_state=0), 
        {
            "ccp_alpha": np.linspace(0, 0.1, 10).astype(int)
        }),

    ('Perceptron CV', Perceptron(random_state=0),
        {
            "alpha": np.linspace(0, 1, 10), 
            "penalty": ["l1", "l2", "elasticnet"]
        }),

    ('RidgeClassifierCV', RidgeClassifierCV(), {})
]