import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mlxtend.evaluate import paired_ttest_5x2cv

acc_dict = {'LR': [],
            'LR_PCA': [],
            'KNN': [],
            'KNN_PCA': [],
            'DT': [],
            'DT_PCA': [],
            'SVC': [],
            'SVC_PCA': [],
            'RF': [],
            'RF_PCA': []}

df = pd.read_csv('odev2.csv') # veriseti okundu

x = df.iloc[:, 0:12] # verisetinin sorular ve etiket kısımları ayrıldı
y = df.iloc[:, 12]
y = y.to_numpy()

ohe = OneHotEncoder()
x = ohe.fit_transform(x).toarray() # kategorik veriler one hot encoding yöntemi ile kodlandı

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42) # %90 eğitim %10 test olacak şekilde veriler bölündü

print('Logistic Regression:')
lr = LogisticRegression()
lr = lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

acc = accuracy_score(y_test, lr_pred)
acc_dict['LR'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', lr_pred[0:10])

print('KNN:')
knn = KNeighborsClassifier()
knn = knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

acc = accuracy_score(y_test, knn_pred)
acc_dict['KNN'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', knn_pred[0:10])

print('Decision Tree')
dt = DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

acc = accuracy_score(y_test, dt_pred)
acc_dict['DT'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', dt_pred[0:10])

print('Support Vector Classifier:')
svc = SVC()
svc = svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)

acc = accuracy_score(y_test, svc_pred)
acc_dict['SVC'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', svc_pred[0:10])

print('Random Forest')
rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

acc = accuracy_score(y_test, rf_pred)
acc_dict['RF'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', rf_pred[0:10])

print('T-Test Results: ')
t, p = paired_ttest_5x2cv(estimator1=lr, estimator2=knn, X=x, y=y)
print('t statistic between lr and knn: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=lr, estimator2=dt, X=x, y=y)
print('t statistic between lr and dt: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=lr, estimator2=svc, X=x, y=y)
print('t statistic between lr and svc: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=lr, estimator2=rf, X=x, y=y)
print('t statistic between lr and rf: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=knn, estimator2=dt, X=x, y=y)
print('t statistic between knn and dt: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=knn, estimator2=svc, X=x, y=y)
print('t statistic between knn and svc: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=knn, estimator2=rf, X=x, y=y)
print('t statistic between knn and rf: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=dt, estimator2=svc, X=x, y=y)
print('t statistic between dt and svc: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=dt, estimator2=rf, X=x, y=y)
print('t statistic between dt and rf: %.3f' % t)

t, p = paired_ttest_5x2cv(estimator1=svc, estimator2=rf, X=x, y=y)
print('t statistic between svc and rf: %.3f' % t)

principal=PCA(n_components=2) # PCA
principal.fit(x)
x_ = principal.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.1, random_state=42)

print('Logistic Regression PCA:')
lr = LogisticRegression()
lr = lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

acc = accuracy_score(y_test, lr_pred)
acc_dict['LR_PCA'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', lr_pred[0:10])

print('KNN PCA:')
knn = KNeighborsClassifier()
knn = knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

acc = accuracy_score(y_test, knn_pred)
acc_dict['KNN_PCA'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', knn_pred[0:10])

print('Decision Tree PCA')
dt = DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

acc = accuracy_score(y_test, dt_pred)
acc_dict['DT_PCA'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', dt_pred[0:10])

print('Support Vector Classifier PCA:')
svc = SVC()
svc = svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)

acc = accuracy_score(y_test, svc_pred)
acc_dict['SVC_PCA'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', svc_pred[0:10])

print('Random Forest PCA')
rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

acc = accuracy_score(y_test, rf_pred)
acc_dict['RF_PCA'].append(acc)

print('Accuracy: ', acc)
print('y_test: ', y_test[0:10])
print('Prediction: ', rf_pred[0:10])

courses = list(acc_dict.keys())
values = [v[0] for v in acc_dict.values()]

colors = ['blue', 'red'] * (len(courses) // 2)

plt.bar(courses, values, color=colors) # PCA uygulanmış ve uygulanmamış verideki doğruluk oranları çizdirildi
plt.show()