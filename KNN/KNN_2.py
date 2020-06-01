import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")


print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot_safety = le.fit_transform(list(data["lug_boot_safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying,maint,door,persons,lug_boot_safety))
y = list(cls)

x_train, x_test, y_train, y_test,  = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

def runner(n):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(x_train, y_train)
    acc = model.score(x_test,y_test)
    print("n:",n," acc:",acc)

#for i in range(1,35):
#    runner(i)


model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test,y_test)
print("acc:",acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    #print("Predicted: ", predicted[x], "Data: ", x_test[x], "Actual: ", y_test[x])
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

print(" ")
print("<end KNN_2>")