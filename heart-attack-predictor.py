from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load Dataset
file_name = "heart.csv"
categories = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
dataset = read_csv(file_name, names=categories)

#Separate into X and Y values
array = dataset.values
x = array[1:,0:-1]
y = array[1:,-1]

#Separate into Training and Validation Data
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size = 0.10, random_state = 1)


#Fit the Model and Make Predictions
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)

#Ask for Patient Info
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

#Convert All Values into Floats
patient = list(map(float, patient))
patient = [patient]

#Make Prediction and Print Result
prediction = model.predict(patient)
if prediction == 1:
    print("This patient has more chance of experiencing a heart attack")
else:
    print("This patient has less chance of experiencing  a heart attack")




