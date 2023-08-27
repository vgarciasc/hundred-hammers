import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Ridge, PassiveAggressiveRegressor, RidgeCV, LassoCV, ElasticNetCV, \
    Perceptron, LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, balanced_accuracy_score

DEFAULT_REGRESSION_METRICS = [
    ('MSE', mean_squared_error, {}),
    ('R2', r2_score, {})
]

DEFAULT_CLASSIFICATION_METRICS = [
    ('Accuracy', accuracy_score, {}),
    # ('Balanced Accuracy', balanced_accuracy_score, {}),
]

DEFAULT_REGRESSION_MODELS = [
    ("Dummy Mean", DummyRegressor(strategy="mean"), {}),
    ("Dummy Median", DummyRegressor(strategy="median"), {}),
    ("Linear Regression", LinearRegression(), {}),
    ("Decision Tree", DecisionTreeRegressor(), {}),
    ("SVR", SVR(), {}),
    ("Linear SVR", LinearSVR(), {}),
    ("Ridge", Ridge(), {}),
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
    ("Decision Tree", DecisionTreeClassifier(random_state=0), {}),
    ('SVC', SVC(gamma='auto', kernel='rbf', ), {}),
    ('Linear SVC', LinearSVC(random_state=0, tol=1e-5), {}),
    ('Perceptron', Perceptron(random_state=0), {}),
    ('Logistic Regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto'), {}),
    ('Ridge Classifier', RidgeClassifier(random_state=0), {}),
    ('SGD Classifier', SGDClassifier(random_state=0), {}),
    ('Passive Aggressive Classifier', PassiveAggressiveClassifier(random_state=0), {}),
    ('K Neighbors Classifier', KNeighborsClassifier(), {}),
    ('MLP Classifier', MLPClassifier(random_state=0), {}),
    ('Gaussian Process Classifier', GaussianProcessClassifier(random_state=0), {}),
    ('Random Forest Classifier', RandomForestClassifier(random_state=0), {}),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=0), {}),
    ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=0), {}),
    ('Extra Trees Classifier', ExtraTreesClassifier(random_state=0), {}),
    ('Bagging Classifier', BaggingClassifier(random_state=0), {}),
    ("Decision Tree CV", DecisionTreeClassifier(random_state=0), {"ccp_alpha": np.linspace(0, 0.1, 10)}),
    ('Perceptron CV', Perceptron(random_state=0),
     {"alpha": np.linspace(0, 1, 10), "penalty": ["l1", "l2", "elasticnet"]}),
    ('RidgeClassifierCV', RidgeClassifierCV(), {}),
    ('MLPClassifier_CV', MLPClassifier(random_state=0),
     {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)], 'activation': ['tanh', 'relu'],
      'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05], 'learning_rate': ['constant', 'adaptive'], }),
]
