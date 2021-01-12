import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score, plot_precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, model_selection, svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier



df = pd.read_csv('/Users/brenda/PycharmProjects/Final_Project_DataScience/final_data/data.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
x = df.filter(['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea', 'Gender'])
y = df['Age']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=21)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(x_train)
X_test_std = stdsc.transform(x_test)



#KNearest Neighbour Model

KNearestNeighbour = KNeighborsClassifier(n_neighbors = 15, weights='uniform')
KNearestNeighbour.fit(x_train, y_train)

y_pred = KNearestNeighbour.predict(x_test)


print("\nKNearest Neighbor\n")

print("Confusion Matrix\n -------------------")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n -----------------------")
print(classification_report(y_test, y_pred))

print("\nPrecision Recall Curve\n -------------------")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(KNearestNeighbour, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))

accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy:", accuracy, "%")



#Support Vector Machine Model

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

y_pred2 = svclassifier.predict(x_test)

print("\nSupport vector Machine\n")

print("Confusion Matrix\n -------------------")
print(confusion_matrix(y_test, y_pred2))

print("\nClassification Report\n -----------------------")
print(classification_report(y_test, y_pred2))

print("\nPrecision Recall Curve\n -------------------")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred2)
average_precision = average_precision_score(y_test, y_pred2)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(KNearestNeighbour, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))

accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy:", accuracy, "%")



#Decision Tree Classifier

dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

dt_entropy.fit(x_train, y_train)

y_pred3= dt_entropy.predict(x_test)


print("\nDecision Tree Classifier\n")

print("Confusion Matrix\n -------------------")
print(confusion_matrix(y_test, y_pred3))

print("\nClassification Report\n -----------------------")
print(classification_report(y_test, y_pred3))

print("\nPrecision Recall Curve\n -------------------")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred3)
average_precision = average_precision_score(y_test, y_pred3)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(KNearestNeighbour, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))

accuracy = accuracy_score(y_test,y_pred3)*100
print("Accuracy:", accuracy, "%")



#Niave Bayes

clf = GaussianNB()
clf.fit(X_train_std, y_train)
GaussianNB(priors=None)
y_pred4 = clf.predict(X_test_std)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

print("\nNaive Bayes\n")

print("Confusion Matrix\n -------------------")
print(confusion_matrix(y_test, y_pred4))

print("\nClassification Report\n -----------------------")
print(classification_report(y_test, y_pred4))

print("\nPrecision Recall Curve\n -------------------")
precision, recall, thresholds = precision_recall_curve(y_test, y_pred4)
average_precision = average_precision_score(y_test, y_pred4)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(KNearestNeighbour, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))

accuracy = accuracy_score(y_test,y_pred4)*100
print("Accuracy:", accuracy, "%")


