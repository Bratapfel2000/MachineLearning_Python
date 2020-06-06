import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(type(cancer))
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data     #1
y = cancer.target   #1


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2 ) #1

#print(x_train, x_test, y_train, y_test)

#print(x_train,y_train)
#print(x_train,y_train)
classes = ['malignant', 'benign']

#clf = svm.SVC()                                      #1
#clf = svm.SVC(kernel="poly")                                #2

#clf = svm.SVC(kernel="linear")                                     #3

# C is soft margin, c= number of points that can pass line
#clf = svm.SVC(kernel="linear", C=3)                                         #4

#compare svm with knn
clf = KNeighborsClassifier(n_neighbors=3)                                          #5


clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)