import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

rcParams['figure.figsize'] = (14, 8)

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

# Load dataset
data = pd.read_csv('creditcard.csv', sep=',')

print(data.head())
print(data.info())

# Check missing values
print(data.isnull().values.any())

# Class distribution
count_classes = data['Class'].value_counts(sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Split dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

print(fraud.shape, normal.shape)

# Describe amount
print(fraud.Amount.describe())
print(normal.Amount.describe())

# Histogram plots
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')

bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins=bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

# Sample dataset
data1 = data.sample(frac=0.1, random_state=RANDOM_SEED)

print(data1.shape)
print(data.shape)

# Fraud vs Valid
Fraud = data1[data1['Class'] == 1]
Valid = data1[data1['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))

print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

# Correlation matrix
corrmat = data1.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20, 20))
sns.heatmap(data1[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Feature selection
columns = data1.columns.tolist()
columns = [c for c in columns if c != "Class"]

target = "Class"

state = np.random.RandomState(RANDOM_SEED)

X = data1[columns]
Y = data1[target]

X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

print(X.shape)
print(Y.shape)

# Define models
classifiers = {
    "Isolation Forest": IsolationForest(
        n_estimators=100,
        max_samples=len(X),
        contamination=outlier_fraction,
        random_state=RANDOM_SEED,
        verbose=0
    ),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        contamination=outlier_fraction
    ),
    "Support Vector Machine": OneClassSVM(
        kernel='rbf',
        degree=3,
        gamma=0.1,
        nu=0.05,
        max_iter=-1
    )
}

n_outliers = len(Fraud)

# Fit models
for clf_name, clf in classifiers.items():
    print("\n" + clf_name)

    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_

    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)

    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Convert predictions: 1 → 0 (normal), -1 → 1 (fraud)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    print("Errors:", n_errors)
    print("Accuracy Score:")
    print(accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred))