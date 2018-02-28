# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:48:02 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
df=pd.read_csv('train.csv',header=0)

subdf=df[['Pclass','Sex','Age']]
y=df.Survived

age=subdf['Age'].fillna(value=subdf.Age.mean())

pclass=pd.get_dummies(subdf['Pclass'],prefix='Pclass')

sex=(subdf['Sex']=='male').astype('int')
X=pd.concat([pclass,age,sex],axis=1)
X.head()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_trian, y_test=train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)
clf=clf.fit(X_train,y_trian)
print("准确率:{:.2f}".format(clf.score(X_test,y_test)))

print(clf.feature_importances_)

import matplotlib.pyplot as plt
feature_importance=clf.feature_importances_
important_features=X_train.columns.values[0::]
feature_importance=100.0*(feature_importance/feature_importance.max())
sorted_idx=np.argsort(feature_importance)[::-1]
pos=np.arange(sorted_idx.shape[0])+.5

plt.title('Feature Importance')
plt.barh(pos,feature_importance[sorted_idx[::-1]],color='r',align='center')
plt.yticks(pos,important_features)
plt.xlabel('Relative Importance')
plt.draw()
plt.show()

from sklearn import cross_validation
scores1=cross_validation.cross_val_score(clf,X,y,cv=10)


from sklearn import metrics
def measure_performance(X,y,clf,show_accuracy=True,
                        show_classification_report=True,
                        show_confusion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")
        
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred),"\n")
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred),"\n")
        
measure_performance(X_test,y_test,clf,show_classification_report=True,show_confusion_matrix=True)