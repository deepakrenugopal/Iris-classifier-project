
# Import evaluation metric
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

iris = load_iris()
X=iris.data
y=iris.target
print(iris.feature_names, iris.target_names)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model= DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Predictions:" , y_pred[:5])
print("True_lables:" , y_test[:5])
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test , y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()