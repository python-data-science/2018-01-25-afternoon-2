import pandas as pd
import numpy as py
import time
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def Elapsed_Time(started):
    later = time.time()

    hours = int((later - started) / 3600)
    minutes = int(((later - started) % 3600) / 60)
    seconds = int(((later - started) % 3600) % 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


df = pd.read_csv("lemon_training.csv")
df = df.drop(["PRIMEUNIT", "AUCGUART"], axis = 1)
df.dropna(axis=0, how='any', inplace=True)
df["ModelNumeric"] = df["Model"].astype("category", ordered = True).cat.codes # Thank you B.Turnwald
df["MakeNumeric"] = df["Make"].astype("category", ordered = True).cat.codes
df = df[[type(d) != float for d in df["Size"]]] # Dump the two Nans in Size
df["SizeNumeric"] = df["Size"].astype("category", ordered = True).cat.codes
df = df[[type(d) != float for d in df["Transmission"]]] # Dump the handful of Nans in Transmission
df["TransmissionNumeric"] = df["Transmission"].astype("category", ordered = True).cat.codes

y = df["IsBadBuy"]
X = df[["VehYear", "ModelNumeric", "MakeNumeric", "TransmissionNumeric", "SizeNumeric", "VehOdo"]]
X.dropna(axis=0, how='any', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = GradientBoostingClassifier().fit(X_train, y_train)
print("SCORE: {}".format(model.score(X_test, y_test)))

cv = KFold(n_splits = 10)

for feature_count in range(3, len(X.columns) + 1):
    for estimator in range (100, 3000, 500):
        start_time = time.time()
        model = GradientBoostingClassifier(max_features = feature_count, n_estimators = estimator)
        scores = []
        for train_i, test_i in cv.split(X):
            Xr, yr, Xt, yt = X.loc[train_i], y.loc[train_i], X.loc[test_i], y.loc[test_i] # Divides the data according to the KSplit
            model.fit(Xr, yr)
            scores.append(model.score(Xt, yt))
        print('features:', feature_count, 'estimators:', estimator, 'scores:', sum(scores)/len(scores), Elapsed_Time(start_time))
