"""
This module provides an easy way of accessing all available machine learning models
and provides some default models to use for classification and regression tasks.
"""

from sklearn.dummy import DummyClassifier # pylint: disable=W0611
from sklearn.dummy import DummyRegressor # pylint: disable=W0611

from sklearn.isotonic import IsotonicRegression # pylint: disable=W0611

from sklearn.neighbors import KNeighborsClassifier # pylint: disable=W0611
from sklearn.neighbors import KNeighborsRegressor # pylint: disable=W0611

from sklearn.linear_model import LinearRegression # pylint: disable=W0611
from sklearn.linear_model import Ridge # pylint: disable=W0611
from sklearn.linear_model import RidgeCV # pylint: disable=W0611
from sklearn.linear_model import Lasso # pylint: disable=W0611
from sklearn.linear_model import LassoCV # pylint: disable=W0611
from sklearn.linear_model import ElasticNet # pylint: disable=W0611
from sklearn.linear_model import ElasticNetCV # pylint: disable=W0611
from sklearn.linear_model import LogisticRegression # pylint: disable=W0611
from sklearn.linear_model import RidgeClassifier # pylint: disable=W0611
from sklearn.linear_model import RidgeClassifierCV # pylint: disable=W0611
from sklearn.linear_model import SGDClassifier # pylint: disable=W0611
from sklearn.linear_model import Perceptron # pylint: disable=W0611
from sklearn.linear_model import PassiveAggressiveClassifier # pylint: disable=W0611
from sklearn.linear_model import PassiveAggressiveRegressor # pylint: disable=W0611

from sklearn.kernel_ridge import KernelRidge# pylint: disable=W0611

from sklearn.neural_network import MLPClassifier # pylint: disable=W0611
from sklearn.neural_network import MLPRegressor # pylint: disable=W0611

from sklearn.svm import LinearSVC # pylint: disable=W0611
from sklearn.svm import LinearSVR # pylint: disable=W0611
from sklearn.svm import SVC # pylint: disable=W0611
from sklearn.svm import SVR # pylint: disable=W0611

from sklearn.tree import DecisionTreeClassifier # pylint: disable=W0611
from sklearn.tree import DecisionTreeRegressor # pylint: disable=W0611
from sklearn.tree import ExtraTreeClassifier # pylint: disable=W0611
from sklearn.tree import ExtraTreeRegressor # pylint: disable=W0611

from sklearn.ensemble import AdaBoostClassifier # pylint: disable=W0611
from sklearn.ensemble import AdaBoostRegressor # pylint: disable=W0611
from sklearn.ensemble import GradientBoostingClassifier # pylint: disable=W0611
from sklearn.ensemble import GradientBoostingRegressor # pylint: disable=W0611
from sklearn.ensemble import RandomForestClassifier # pylint: disable=W0611
from sklearn.ensemble import RandomForestRegressor # pylint: disable=W0611
from sklearn.ensemble import ExtraTreesClassifier # pylint: disable=W0611
from sklearn.ensemble import BaggingClassifier # pylint: disable=W0611

from sklearn.naive_bayes import BernoulliNB # pylint: disable=W0611
from sklearn.naive_bayes import CategoricalNB # pylint: disable=W0611
from sklearn.naive_bayes import ComplementNB # pylint: disable=W0611
from sklearn.naive_bayes import GaussianNB # pylint: disable=W0611
from sklearn.naive_bayes import MultinomialNB # pylint: disable=W0611

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # pylint: disable=W0611
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis # pylint: disable=W0611

from sklearn.gaussian_process import GaussianProcessClassifier # pylint: disable=W0611
from sklearn.gaussian_process import GaussianProcessRegressor # pylint: disable=W0611

# XGBoost library
from xgboost import XGBRegressor # pylint: disable=W0611
from xgboost import XGBClassifier # pylint: disable=W0611

# scikit-elm library
from skelm import ELMRegressor # pylint: disable=W0611
from skelm import ELMClassifier # pylint: disable=W0611

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
    ("Decision Tree", DecisionTreeRegressor(), {}),
    ("SVR", SVR(), {}),
    ("Linear SVR", LinearSVR(dual="auto"), {}),
    ("Ridge", Ridge(),{}),
    ("Passive Aggressive", PassiveAggressiveRegressor(), {}),
    ("KNN", KNeighborsRegressor(), {}),
    ("Neural Network Regressor", MLPRegressor(), {}),
    ("Gaussian Process", GaussianProcessRegressor(), {}),
    ("Random Forest", RandomForestRegressor(), {}),
    ("AdaBoost", AdaBoostRegressor(), {}),
    ("Gradient Boosting", GradientBoostingRegressor(), {})
]

DEFAULT_CLASSIFICATION_MODELS = [
    ("Dummy", DummyClassifier(strategy="most_frequent"), {}),
    ("Decision Tree", DecisionTreeClassifier(random_state=0), {}),
    ('SVC', SVC(gamma='auto', kernel='rbf'), {}),
    ('Linear SVC', LinearSVC(random_state=0, tol=1e-5, dual="auto"), {}),
    ('Perceptron', Perceptron(random_state=0), {}),
    ('Logistic Regression', LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto'), {}),
    ('Ridge Classifier', RidgeClassifier(random_state=0), {}),
    ('SGD Classifier', SGDClassifier(random_state=0), {}),
    ('Passive Aggressive Classifier', PassiveAggressiveClassifier(random_state=0), {}),
    ('K Neighbors Classifier', KNeighborsClassifier(), {}),
    ('Neural Network Classifier', MLPClassifier(random_state=0), {}),
    ('Gaussian Process Classifier', GaussianProcessClassifier(random_state=0), {}),
    ('Random Forest Classifier', RandomForestClassifier(random_state=0), {}),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=0), {}),
    ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=0), {}),
    ('Extra Trees Classifier', ExtraTreesClassifier(random_state=0), {}),
    ('Bagging Classifier', BaggingClassifier(random_state=0), {})
]