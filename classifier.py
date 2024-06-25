import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from ydata_profiling import ProfileReport
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="My report")
# profile.to_file("diabetes_report.html")
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=recall_score)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# params = {
#     "n_estimators": [50, 100, 200],
#     "criterion": ["gini", "entropy", "log_loss"],
#     "max_depth": [None, 2, 5]
# }
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(),
#     param_grid=params,
#     scoring="recall",
#     cv=6,
#     n_jobs=-1,
#     verbose=1
# )
# model = LogisticRegression()
# model = RandomForestClassifier()
# grid_search.fit(x_train, y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)
# #
# y_predict = grid_search.predict(x_test)
# print(classification_report(y_test, y_predict))



