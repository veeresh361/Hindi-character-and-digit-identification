#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


d=pd.read_csv('loan_borowwer_data.csv',header=0,sep=',',encoding='latin')
d.head(10)


# In[3]:


d.shape


# In[4]:


d.isnull().sum()


# In[5]:


d['credit.policy'].value_counts()


# In[6]:


plt.figure(figsize=(20,20))
sns.heatmap(d.corr(),annot=True,cmap='YlGnBu')


# In[8]:


plt.figure(figsize=(20,20))
sns.heatmap(d.cov(),annot=True,cmap='YlGnBu')


# In[7]:


x=d.drop(columns=['not.fully.paid'],axis=1)
y=d['not.fully.paid']
x=pd.get_dummies(x)


# In[8]:


x.shape


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[10]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[30]:


x_train


# In[11]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=100,penalty='l2')
lr.fit(x_train,y_train)


# In[12]:


e=lr.predict_proba(x_test)


# In[13]:


import sklearn.metrics as metrics
metrics.roc_curve(y_test,e[:,1])


# In[14]:


fpr,tpr,thresholds=metrics.roc_curve(y_test,e[:,1])
plt.plot(fpr,tpr,'-')


# In[15]:


lr.classes_


# In[16]:


y_pred=lr.predict(x_test)
y_a=lr.predict_proba(x_test)[:,1]
from sklearn.metrics import accuracy_score ,roc_auc_score
acc=accuracy_score(y_test,y_pred)
ar=roc_auc_score(y_test,y_a)
results=pd.DataFrame([['Logistic Regression',acc,ar]],columns=['Model','Accuracy','Area Under Curve'])
results


# In[17]:


from sklearn.tree import DecisionTreeClassifier
dtr=DecisionTreeClassifier(criterion='entropy',random_state=100)
dtr.fit(x_train,y_train)


# In[18]:


y_pred=dtr.predict(x_test)
y_a=dtr.predict_proba(x_test)[:,1]
from sklearn.metrics import accuracy_score ,roc_auc_score
acc=accuracy_score(y_test,y_pred)
ar=roc_auc_score(y_test,y_a)
result1=pd.DataFrame([['DecicsionTree',acc,ar]],columns=['Model','Accuracy','Area Under Curve'])
results=results.append(result1)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=100)
rf.fit(x_train,y_train)


# In[20]:


rf.oob_score_


# In[22]:


for w in range(10,300,20):
    rf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=100)
    rf.fit(x_train,y_train)
    oob=rf.oob_score_
    print('For n_estimators='+str(w))
    print('oob score is '+str(oob))
    print('**********************')
    


# In[21]:


rf=RandomForestClassifier(n_estimators=290,oob_score=True,n_jobs=-1,random_state=100)
rf.fit(x_train,y_train)
rf.oob_score_


# In[22]:


y_pred=rf.predict(x_test)
y_a=rf.predict_proba(x_test)[:,1]
from sklearn.metrics import accuracy_score ,roc_auc_score
acc=accuracy_score(y_test,y_pred)
ar=roc_auc_score(y_test,y_a)
result2=pd.DataFrame([['RandomForestClassififer',acc,ar]],columns=['Model','Accuracy','Area Under Curve'])
results=results.append(result2)
#results=results.append(result1)
results


# In[23]:


sns.barplot(x='Model',y='Accuracy',data=results)

