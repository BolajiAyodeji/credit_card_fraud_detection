import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored as cl
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle

#Import dataset

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.drop('Time', axis = 1, inplace = True)

print(df.head())

#Process data and some Exploratory Data Analysis (EDA)

#Percentage of fraud cases
cases = len(df)
nonfraud_count = len(df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print(cl('CASE COUNT', attrs = ['bold']))
print(cl('--------------------------------------', attrs = ['bold']))
print(cl('Total number of cases are {}'.format(cases)))
print(cl('Number of Non-fraud cases are {}'.format(nonfraud_count)))
print(cl('Number of fraud cases are {}'.format(fraud_count)))
print(cl('Percentage of fraud cases is {}'.format(fraud_percentage)))
print(cl('--------------------------------------', attrs = ['bold']))

#Statistical view of both fraud and non-fraud transaction amount data
nonfraud_cases = df[df.Class == 0]
fraud_cases = df[df.Class == 1]

print(cl('CASE AMOUNT STATISTICS', attrs = ['bold']))
print(cl('|'))
print(cl('----------------------------------', attrs = ['bold']))
print(cl('NON-FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(nonfraud_cases.Amount.describe())
print(cl('----------------------------------'))
print(cl('FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(fraud_cases.Amount.describe())
print(cl('----------------------------------'))

#Normalize the Amount variable
sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

print(cl(df['Amount'].head(10), attrs = ['bold']))

#Split the dataset into train and test data
X = df.drop('Class', axis = 1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(cl('X_train samples : ', attrs = ['bold']), X_train[:1])
print(cl('X_test samples : ', attrs = ['bold']), X_test[0:1])
print(cl('y_train samples : ', attrs = ['bold']), y_train[0:20])
print(cl('y_test samples : ', attrs = ['bold']), y_test[0:20])

#Testing the model using six classification models

#Decision Tree
tree_model = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)

#K-Nearest Neighbors
n = 5

knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

#SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)

#Random Forest Tree
rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

#XGBoost
xgb = XGBClassifier(max_depth = 4, use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

#Models evaluation

#Accuracy score
print(cl('ACCURACY SCORE', attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, tree_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the KNN model is {}'.format(accuracy_score(y_test, knn_yhat)), color = 'green'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)), color = 'red'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the SVM model is {}'.format(accuracy_score(y_test, svm_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the Random Forest Tree model is {}'.format(accuracy_score(y_test, rf_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))

#F1 score
print(cl('F1 SCORE', attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, tree_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the KNN model is {}'.format(f1_score(y_test, knn_yhat)), color = 'green'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)), color = 'red'))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the SVM model is {}'.format(f1_score(y_test, svm_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the Random Forest Tree model is {}'.format(f1_score(y_test, rf_yhat))))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat))))
print(cl('---------------------------', attrs = ['bold']))

#Confusion Matrix
def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix for the models
tree_matrix = confusion_matrix(y_test, tree_yhat, labels = [0, 1])
knn_matrix = confusion_matrix(y_test, knn_yhat, labels = [0, 1])
lr_matrix = confusion_matrix(y_test, lr_yhat, labels = [0, 1])
svm_matrix = confusion_matrix(y_test, svm_yhat, labels = [0, 1])
rf_matrix = confusion_matrix(y_test, rf_yhat, labels = [0, 1])
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels = [0, 1])

#Plot the confusion matrix
plt.rcParams['figure.figsize'] = (6, 6)

#Confusion matrix for Decision tree
tree_cm_plot = plot_confusion_matrix(tree_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'Decision Tree')

plt.savefig('tree_cm_plot.png')
plt.show()

#Confusion matrix for K-Nearest Neighbors
knn_cm_plot = plot_confusion_matrix(knn_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'K-Nearest Neighbors')

plt.savefig('knn_cm_plot.png')
plt.show()

#Confusion matrix for Logistic regression
lr_cm_plot = plot_confusion_matrix(lr_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'Logistic Regression')

plt.savefig('lr_cm_plot.png')
plt.show()

#Confusion matrix for Support Vector Machine
svm_cm_plot = plot_confusion_matrix(svm_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'SVM')

plt.savefig('svm_cm_plot.png')
plt.show()

#Confusion matrix for Random forest tree
rf_cm_plot = plot_confusion_matrix(rf_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'Random Forest Tree')

plt.savefig('rf_cm_plot.png')
plt.show()

#Confusion matrix for XGBoost
xgb_cm_plot = plot_confusion_matrix(xgb_matrix, 
classes = ['Non-Default(0)','Default(1)'], 
normalize = False, title = 'XGBoost')

plt.savefig('xgb_cm_plot.png')
plt.show()

#Test model with new transactions
trans = [[0.32333357, 1.05745525, 1.04834115, 0.60720431, 1.25982115, 1.09176072,
   1.1591015, 1.12433461, 1.17463954, 1.64440065, 0.11886302, 1.20264731,
   1.14596495, 1.80235956, 1.24717793, 1.06094535, 1.84660574, 1.37945439,
   1.84726224, -0.18640942, 1.20709827, 0.43389027, 1.26161328, 1.04665061,
   1.2115123, 1.00829721, 1.10849443, 1.16113917, -0.19330595]]

result = knn.predict(trans)
print(result)

#Save model in SAV format
filename = 'credit_card_fraud_detection.sav'
pickle.dump(knn, open(filename, 'wb'))

#Load saved model and test

filename = './credit_card_fraud_detection.sav'
loaded_model = pickle.load(open(filename, 'rb'))

trans = [[-0.32333357, 1.05745525, -0.04834115, -0.60720431, 1.25982115, -0.09176072,
   1.1591015, -0.12433461, -0.17463954, -1.64440065, -1.11886302, 0.20264731,
   1.14596495, -1.80235956, -0.24717793, -0.06094535, 0.84660574, 0.37945439,
   0.84726224, 0.18640942, -0.20709827, -0.43389027, -0.26161328, -0.04665061,
   0.2115123, 0.00829721, 0.10849443, 0.16113917, -0.19330595]]

result = loaded_model.predict(trans)
print(result)