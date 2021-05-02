import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file_name = "heart.csv"
names = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
dataset = read_csv(file_name, names=names)

print(dataset.shape)

print(dataset.head(20))

print(dataset.describe())
print(dataset.groupby('output').size())



array = dataset.values
x = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size = 0.20, random_state = 1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
#pyplot.show()

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(predictions)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

patient = []
patient.append(input("Enter Age of the Patient"))
patient.append(input("Sex:"))
patient.append(input("cp:"))
patient.append(input("trtbps:"))
patient.append(input("chol:"))
patient.append(input("fbs:"))
patient.append(input("restecg:"))
patient.append(input("thatlachh:"))
patient.append(input("exng:"))
patient.append(input("oldpeak:"))
patient.append(input("slp:"))
patient.append(input("caa:"))
patient.append(input("thall:"))

patient = list(map(float, patient))
patient = [patient]

prediction = model.predict(patient)
if prediction = 1:
    print("This patient has more chance of experiencing a heart attack")
else:
    print("This patient has less chance of experiencing  a heart attack")
#print(x)
#print(y)



