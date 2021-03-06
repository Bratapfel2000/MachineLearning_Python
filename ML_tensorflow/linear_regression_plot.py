import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")


data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test,  = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

"""
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test,  = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print("acc >",acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

pickle_in =  open("studentmodel.pickle", "rb")
print(pickle_in)
#print(50*"=")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
    if ((x_test[x][0]+x_test[x][1])/2)-y_test[x]>3:
        print(predictions[x], x_test[x], y_test[x], 30*"<", "deviance")

#p = "G1"
#p = "G2"
#p = "studytime"
#p = "failures"
p = "absences"

style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


print("-<end>-")