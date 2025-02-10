from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from lib import util
import pandas as pd

classification_datasets = ["german-credit", "adult", "breast-cancer"]
regression_datasets = ["california-housing", "abalone", "insurance", "ACS2018"]
dp_datasets = ["adult-reduced"]

def check_dataset_name(dataset):
    if (dataset not in classification_datasets) and (dataset not in regression_datasets) and (dataset not in dp_datasets):
        raise Exception("Unknown dataset {}".format(dataset))

def is_classification(dataset):
    return dataset in classification_datasets

target_names = {
    "german-credit": "TargetGood",
    "adult": "income",
    "adult-reduced": "income",
    "breast-cancer": "target",

    "california-housing": "LogMedHouseVal",
    "abalone": "Rings",
    "insurance": "log_charges",
    "ACS2018": "log_PINCP",
}

classification_algos = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "1-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    "5-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000)),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
}
regression_algos = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge()),
    "1-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1)),
    "5-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000)),
    "SVM": make_pipeline(StandardScaler(), SVR()),
}

# algos_per_dataset = {
#     dataset: classification_algos if dataset in classification_datasets else regression_algos
#     for dataset in classification_datasets + regression_datasets
# }

n_generated_syn_datasets = 5

def adult_reduced_discrete_to_continuous(df):
    c_df = df.copy()
    df["age"] = df.age.apply(lambda age: util.interval_middle(age)).astype(float)
    df["hours_per_week"] = df["hours_per_week"].apply(lambda hours: util.interval_middle(hours)).astype(float)
    return df

def adult_reduced_discretisation(df):
    df = df.copy()
    df.age = pd.cut(df.age, [0, 20, 40, 60, 80, 100])
    df["hours_per_week"] = pd.cut(df["hours_per_week"], [0, 25, 35, 45, 55, 100])
    return df

napsu_query_selection_epsilon = 0.5
napsu_measurement_epsilon = 1.0
total_epsilon = napsu_query_selection_epsilon + napsu_measurement_epsilon

total_delta = 46043**(-2)
napsu_query_selection_delta = 0.5 * total_delta
napsu_measurement_delta = 0.5 * total_delta