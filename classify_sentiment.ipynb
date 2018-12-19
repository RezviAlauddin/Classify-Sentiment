
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

import pandas as pd
from pandas import Series
import numpy as np
import csv
import matplotlib.pyplot as plt




df = pd.read_csv('train.csv',encoding='utf-8')
vec = CountVectorizer()



dft = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]


hate=[]
for index,row in dft.iterrows():
    hate.append(row.toxic + row.severe_toxic + row.obscene + row.threat + row.insult + row.identity_hate)
    
df['hate'] = hate



df['hate'] = (df['hate'] > 0).astype(int)


X = df[['comment_text']]
Y = df[['hate']]


x_train, x_test, y_train, y_test = train_test_split(X['comment_text'],Y['hate'],test_size=0.2,random_state=1)




vec.fit(X['comment_text'])

x_train = vec.transform(x_train)

x_test = vec.transform(x_test)





A=[]
print("Give any five sentances below: ")
for i in range(5):
    A.append(str(input()))

x_try = vec.transform(A)





NB = MultinomialNB(alpha=1.5)
NB.fit(x_train,y_train)
NB_predicted = NB.predict(x_test)
NBC = NB.score(x_test,y_test)


pred = NB.predict(x_try)

z = len(pred)
for i in range(z):
    if pred[i]==0:
        print(i+1," no. sentance is not hatefull")
    else:
        print(i+1," no. sentance is hatefull")


tn,fp,fn,tp=confusion_matrix(y_test,NB_predicted).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Naive Bayes(Multinominal) Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Naive Bayes Classifier: ",NBC*100,"%\n")
print("Error of Naive Bayes Classifier: ",(1-NBC)*100,"%\n")
print("Precision of Naive Bayes Classifier: ",(tp/(tp+fp))*100,"%\n")
print("Recall of Naive Bayes Classifier: ",(tp/(tp+fn))*100,"%\n")

NBP = tp/(tp+fp)
NBR = tp/(tp+fn)




LR = LogisticRegression()
LR.fit(x_train,y_train)
LR_predicted = LR.predict(x_test)
LRC = LR.score(x_test,y_test)

pred = LR.predict(x_try)
z = len(pred)
for i in range(z):
    if pred[i]==0:
        print(i+1," no. sentance is not hatefull")
    else:
        print(i+1," no. sentance is hatefull")


tn,fp,fn,tp=confusion_matrix(y_test,LR_predicted).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Logistic Regression Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Logistic Regression Classifier: ",LRC*100,"%\n")
print("Error of Logistic Regression Classifier: ",(1-LRC)*100,"%\n")
print("Precision of Logistic Regression Classifier: ",(tp/(tp+fp))*100,"%\n")
print("Recall of Logistic Regression Classifier: ",(tp/(tp+fn))*100,"%\n")

LRP = tp/(tp+fp)
LRR = tp/(tp+fn)




S = LinearSVC()
S.fit(x_train,y_train)
S_predicted = S.predict(x_test)
SC = S.score(x_test,y_test)


pred = S.predict(x_try)
z = len(pred)
for i in range(z):
    if pred[i]==0:
        print(i+1," no. sentance is not hatefull")
    else:
        print(i+1," no. sentance is hatefull")

tn,fp,fn,tp=confusion_matrix(y_test,S_predicted).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Linear Support Vector  Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Linear Support Vector Classifier: ",SC*100,"%\n")
print("Error of Linear Support Vector Classifier: ",(1-SC)*100,"%\n")
print("Precision of Linear Support Vector Classifier: ",(tp/(tp+fp))*100,"%\n")
print("Recall of Linear Support Vector Classifier: ",(tp/(tp+fn))*100,"%\n")

SCP = tp/(tp+fp)
SCR = tp/(tp+fn)




DT = DecisionTreeClassifier(max_depth=500)
DT.fit(x_train,y_train)
DT_predicted = DT.predict(x_test)
DTC = DT.score(x_test,y_test)


pred = DT.predict(x_try)
z = len(pred)
for i in range(z):
    if pred[i]==0:
        print(i+1," no. sentance is not hatefull")
    else:
        print(i+1," no. sentance is hatefull")

tn,fp,fn,tp=confusion_matrix(y_test,DT_predicted).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Decision Tree Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Decision Tree Classifier: ",DTC*100,"%\n")
print("Error of Decision Tree Classifier: ",(1-DTC)*100,"%\n")
print("Precision of Decision Tree Classifier: ",(tp/(tp+fp))*100,"%\n")
print("Recall of Decision Tree Classifier: ",(tp/(tp+fn))*100,"%\n")

DTP = tp/(tp+fp)
DTR = tp/(tp+fn)




RF = RandomForestClassifier(max_depth=500)
RF.fit(x_train,y_train)
RF_predicted = RF.predict(x_test)
RFC = RF.score(x_test,y_test)


pred = RF.predict(x_try)
z = len(pred)
for i in range(z):
    if pred[i]==0:
        print(i+1," no. sentance is not hatefull")
    else:
        print(i+1," no. sentance is hatefull")

tn,fp,fn,tp=confusion_matrix(y_test,RF_predicted).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Random Forest Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Random Forest Classifier: ",RFC*100,"%\n")
print("Error of Random Forest Classifier: ",(1-RFC)*100,"%\n")
print("Precision of Random Forest Classifier: ",(tp/(tp+fp))*100,"%\n")
print("Recall of Random Forest Classifier: ",(tp/(tp+fn))*100,"%\n")

RFP = tp/(tp+fp)
RFR = tp/(tp+fn)




plt.hist([[NBC],[LRC],[SC],[DTC],[RFC]], bins=2, color=['red','blue','yellow','green','orange'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title('Accuracy of different classifiers')
plt.show()




X = [NBC,LRC,SC,DTC,RFC]
plt.hist(X)
plt.ylabel=("Accuracy")
plt.xlabel=("NBC","LRC","SC","DTC","RFC")
plt.show()




X = [1-NBC,1-LRC,1-SC,1-DTC,1-RFC]
plt.hist(X)
plt.ylabel=("Error")
plt.xlabel=("NBC","LRC","SC","DTC","RFC")
plt.show()




X = [NBP,LRP,SCP,DTP,RFP]
plt.hist(X)
plt.ylabel=("Precision")
plt.xlabel=("NBC","LRC","SC","DTC","RFC")
plt.show()




X = [NBR,LRR,SCR,DTR,RFR]
plt.hist(X)
plt.ylabel=("Recall")
plt.xlabel=("NBC","LRC","SC","DTC","RFC")
plt.show()

