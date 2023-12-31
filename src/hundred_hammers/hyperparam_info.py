from schema import (
    Schema,
    Or,
    Optional,
    Regex,
)

hyperparam_def_schema = Schema(
    {
        "model": str,
        Optional(Regex(".*")): Or(
            {"type": "real", "min": Or(float, int), "max": Or(float, int)},
            {"type": "integer", "min": int, "max": int},
            {
                "type": "categorical",
                "values": [lambda x: True],
            },  # Accept a list of any type
        ),
    }
)

known_hyperparams = [
    {
        "model": "DummyClassifier",
        "strategy": {
            "type": "categorical",
            "values": ["most_frequent", "prior", "stratified", "uniform"],
        },
    },
    {
        "model": "DummyRegressor",
        "strategy": {"type": "categorical", "values": ["mean", "median"]},
    },
    {"model": "IsotonicRegression"},
    {
        "model": "KNeighborsClassifier",
        "n_neighbors": {"type": "integer", "min": 1, "max": 500},
        "metric": {
            "type": "categorical",
            "values": ["manhattan", "euclidean", "cosine"],
        },
    },
    {
        "model": "KNeighborsRegressor",
        "n_neighbors": {"type": "integer", "min": 1, "max": 500},
        "metric": {
            "type": "categorical",
            "values": ["manhattan", "euclidean", "cosine"],
        },
    },
    {"model": "LinearRegression"},
    {"model": "Ridge", "alpha": {"type": "real", "min": 0, "max": 100}},
    {"model": "Lasso", "alpha": {"type": "real", "min": 0, "max": 100}},
    {
        "model": "ElasticNet",
        "alpha": {"type": "real", "min": 0, "max": 100},
        "l1_ratio": {"type": "real", "min": 0, "max": 1},
    },
    {"model": "LogisticRegression"},
    {"model": "RidgeClassifier", "alpha": {"type": "real", "min": 0, "max": 100}},
    {
        "model": "SGDClassifier",
        "penalty": {
            "type": "categorical",
            "values": ["l1", "l2", "elasticnet", "None"],
        },
        "alpha": {"type": "real", "min": 0, "max": 100},
        "l1_ratio": {"type": "real", "min": 0, "max": 1},
    },
    {
        "model": "Perceptron",
        "penalty": {
            "type": "categorical",
            "values": ["l1", "l2", "elasticnet", "None"],
        },
        "alpha": {"type": "real", "min": 0, "max": 100},
        "l1_ratio": {"type": "real", "min": 0, "max": 1},
    },
    {
        "model": "PassiveAggressiveClassifier",
        "C": {"type": "real", "min": 0, "max": 100},
    },
    {
        "model": "KernelRidge",
        "alpha": {"type": "real", "min": 0, "max": 100},
        "kernel": {"type": "categorical", "values": ["linear", "poly", "rbf"]},
    },
    {
        "model": "SVC",
        "C": {"type": "real", "min": 0, "max": 100},
        "kernel": {"type": "categorical", "values": ["linear", "poly", "rbf"]},
        "degree": {"type": "integer", "min": 1, "max": 6},
    },
    {
        "model": "SVR",
        "C": {"type": "real", "min": 0, "max": 100},
        "kernel": {"type": "categorical", "values": ["linear", "poly", "rbf"]},
        "degree": {"type": "integer", "min": 1, "max": 6},
    },
    {
        "model": "DecisionTreeClassifier",
        "criterion": {"type": "categorical", "values": ["gini", "entropy", "log_loss"]},
        "max_depth": {"type": "integer", "min": 1, "max": 20},
    },
    {
        "model": "DecisionTreeRegressor",
        "criterion": {
            "type": "categorical",
            "values": ["friedman_mse", "squared_error", "absolute_error", "poisson"],
        },
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {
        "model": "AdaBoostClassifier",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
    },
    {
        "model": "AdaBoostRegressor",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
    },
    {
        "model": "GradientBoostingClassifier",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
        "criterion": {
            "type": "categorical",
            "values": ["friedman_mse", "squared_error"],
        },
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {
        "model": "GradientBoostingRegressor",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
        "criterion": {
            "type": "categorical",
            "values": ["friedman_mse", "squared_error"],
        },
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {
        "model": "RandomForestClassifier",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
        "criterion": {"type": "categorical", "values": ["gini", "entropy", "log_loss"]},
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {
        "model": "RandomForestRegressor",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
        "criterion": {
            "type": "categorical",
            "values": ["friedman_mse", "squared_error", "absolute_error", "poisson"],
        },
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {"model": "BernoulliNB", "alpha": {"type": "real", "min": 0, "max": 1}},
    {"model": "GaussianNB", "var_smoothing": {"type": "real", "min": 0, "max": 1}},
    {
        "model": "LinearDiscriminantAnalysis",
        "solver": {"type": "categorical", "values": ["svd", "lsqr", "eigen"]},
    },
    {"model": "QuadraticDiscriminantAnalysis"},
    {
        "model": "XGBRegressor",
        "n_estimators": {"type": "integer", "min": 1, "max": 400},
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {
        "model": "XGBClassifier",
        "n_estimators": {"type": "integer", "min": 1, "max": 100},
        "max_depth": {"type": "integer", "min": 1, "max": 30},
    },
    {"model": "ELMRegressor", "n_neurons": {"type": "integer", "min": 1, "max": 400}},
    {"model": "ELMClassifier", "n_neurons": {"type": "integer", "min": 1, "max": 400}},
]

known_models = [k["model"] for k in known_hyperparams]
