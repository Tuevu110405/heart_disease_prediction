import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

#bai tap 1
df = pd.read_csv('cleveland.csv', header = None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
#visualizing distribution of target vs age
plt.figure(figsize=(12, 6))
df.boxplot(column=['target','age'])
plt.xlabel('Variables')
plt.ylabel('Value')
plt.title('Distribution of target vs age')
plt.show()
#visualizing relationship distribution of target and age
plt.figure(figsize=(12, 6))
sns.violinplot(x='target', y='age', data=df)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship distribution of target and age')
plt.show()

#barplot visualizes the relationship of age with hue = target
plt.figure(figsize=(20, 12))
sns.catplot(x='age', hue='target', kind='count', data=df)
plt.show()

plt.figure(figsize=(20, 12))
sns.catplot(x='sex', hue='target',y='age', kind='bar', data=df)
plt.show()

#bai tap 3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_train = 'Accuracy for train:'
acc_test = 'Accuracy for test:'


print('KNN scores:')
#accuracy for train
y_pred_train = knn.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(acc_train, accuracy_train)
#accuracy for test
y_pred_test = knn.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(acc_test, accuracy_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

svm = SVC(kernel='rbf', random_state = 42)
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)
y_train_pred = svm.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_test_pred)

print('Scores for SVC:')
print(acc_train, accuracy_train)
accuracy = accuracy_score(y_test, y_test_pred)
print(acc_test, accuracy)

from sklearn.naive_bayes import GaussianNB

gauss = GaussianNB()
gauss.fit(X_train, y_train)

y_pred_train = gauss.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
y_pred_test = gauss.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print('Scores for Naive Bayes:')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini', max_depth=10, min_samples_split=2, )
dt.fit(X_train, y_train)
y_train_pred = dt.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
y_test_pred = dt.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print('Scores for Decision Tree:')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = 'gini', max_depth=10, min_samples_split=2, n_estimators = 10, random_state= 42)

rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
y_test_pred = rf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print('Scores for randomforest:')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate = 1.0)
ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
y_test_pred = ada.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print('Scores for adaboost:')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth= 3, random_state=42)
gb.fit(X_train, y_train)
y_train_pred = gb.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
y_test_pred = gb.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print('Scores for gradientboosting')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

from xgboost import XGBClassifier
xgb = XGBClassifier(objective = "binary:logistic", random_state=42, n_estimators = 100)
xgb.fit(X_train, y_train)
y_train_pred = xgb.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
y_test_pred = xgb.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print('Scores for xgboost')
print(acc_train, accuracy_train)
print(acc_test, accuracy_test)

#bai tap 3
from sklearn.ensemble import StackingClassifier


dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
svm = SVC(kernel = 'rbf', random_state=42)

ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)


#creating base models
base_models = [('knn', knn), ('svm', svm), ('dt', dt), ('rf', rf), ('ada', ada), ('gb', gb)]

#creating meta model
meta_model = XGBClassifier()

#Tạo mô hình stacking
from sklearn.ensemble import StackingClassifier
stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model)

#Huan luyen mo hinh stacking
stacking.fit(X_train, y_train)

print('Stacking scores:')


#Du doan du lieu train
y_pred_train = stacking.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(acc_train, accuracy_train)

#Du doan du lieu test
y_pred = stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(acc_test, accuracy)


