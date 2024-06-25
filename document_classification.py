import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTEN


def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) > 0:
        return result[0][2:]
    else:
        return location


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
target = "career_level"
data["location"] = data["location"].apply(filter_location)
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

# ros = SMOTEN(random_state=42,
#                         sampling_strategy={"managing_director_small_medium_company": 100, "specialist": 100,
#                                            "director_business_unit_leader": 100, "bereichsleiter": 1000}, k_neighbors=2)
# print(y_train.value_counts())
# print("---------------")
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title_features", TfidfVectorizer(), "title"),
    ("location_features", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    (
    "des_features", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_features", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_features", TfidfVectorizer(), "industry"),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    # ("feature_selection", SelectKBest(chi2, k=300)),
    ("feature_selection", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier()),
])
params = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selection__percentile": [5, 10, 20],
    "preprocessor__des_features__min_df": [0.01, 0.02, 0.05],
    "preprocessor__des_features__max_df": [0.90, 0.95, 0.99],
    "preprocessor__des_features__ngram_range": [(1, 1), (1, 2)],
}
grid_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    scoring="f1_weighted",
    cv=5,
    n_jobs=6,
    verbose=1,
    n_iter=50
)
grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(classification_report(y_test, y_predict))
# 8000 weighted avg       0.71      0.74      0.69      1615
# 800 weighted avg       0.75      0.76      0.73      1615
# 500 weighted avg       0.75      0.76      0.73      1615
# 300 weighted avg       0.77      0.77      0.75      1615


