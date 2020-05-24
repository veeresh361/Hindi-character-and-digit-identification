#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import pandas as pd
import seaborn as sns


# In[130]:


d=pd.read_csv("Churn_Modelling.csv",sep=',',header=0,encoding='latin')


# In[77]:


d.head()


# In[131]:


d=d.drop(['CustomerId','Surname','RowNumber'],axis=1)


# In[132]:


f={'Female':0,'Male':1}
d['Gender']=d['Gender'].map(f)


# In[133]:


d.head()
d=d.drop(['Geography'],axis=1)


# In[43]:


d.head()


# In[134]:


x=d.drop('Exited',axis=1)
y=d['Exited']


# In[135]:


x.head()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[136]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[137]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[86]:


x_test[2]


# In[49]:


x_train=pd.DataFrame(x_train,columns=x.columns)
x_train.head()


# In[50]:


w=x_train
w['Exited']=y_train


# In[15]:


w.head()


# In[87]:


sns.pairplot(w,x_vars=['EstimatedSalary','CreditScore'],y_vars='Exited',kind='reg')


# In[88]:


learning_rate=0.001
x_train=np.array(x_train,np.float32)
x_train.shape
y_train=np.array(y_train)


# In[100]:


x_test=np.array(x_test,np.float32)


# In[54]:


import tensorflow as tf


# In[91]:


random=tf.initializers.RandomNormal()
weights={'h1':tf.Variable(random([9,5])),
        'h2':tf.Variable(random([5,3])),
        'out':tf.Variable(random([3,2]))}
biases={'b1':tf.Variable(tf.zeros([5])),
       'b2':tf.Variable(tf.zeros([3])),
       'out':tf.Variable(tf.zeros([2]))}


# In[92]:


def neural_net(inputdata):
    hidden1=tf.add(tf.matmul(inputdata,weights['h1']),biases['b1'])
    hidden1=tf.nn.sigmoid(hidden1)
    hidden2=tf.add(tf.matmul(hidden1,weights['h2']),biases['b2'])
    hidden2=tf.nn.sigmoid(hidden2)
    output=tf.matmul(hidden2,weights['out'])+biases['out']
    return tf.nn.softmax(output)


# In[57]:


def cross_entropy(y_pred,y_true):
    y_true=tf.one_hot(y_true,depth=2)
    y_pred=tf.clip_by_value(y_pred,1e-9,1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))


# In[93]:


optimizer=tf.keras.optimizers.SGD(learning_rate)
def run_optimization(x,y):
    with tf.GradientTape()as g:
        pred=neural_net(x)
        loss=cross_entropy(pred,y)
        trainable_variables=list(weights.values())+list(biases.values())
        gradients=g.gradient(loss,trainable_variables)
        optimizer.apply_gradients(zip(gradients,trainable_variables))


# In[94]:


def accuracy(y_pred,y_true):
    correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),axis=-1)


# In[95]:


learning_rate=0.001
training_steps=3000
batch_size=250
display_step=100
train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.shuffle(60000).batch(batch_size).prefetch(1)


# In[97]:


for step,(batch_x,batch_y) in enumerate(train_data.take(300),1):
    run_optimization(batch_x,batch_y)
    if (step %display_step)!=0:
        pred=neural_net(batch_x)
        loss=cross_entropy(pred,batch_y)
        acc=accuracy(pred,batch_y)
        print('training epoch:%i,loss:%f,accuracy:%f'%(step,loss,acc))


# In[104]:


pred=neural_net(x_test)
print('accuracy:%f' % accuracy(pred,y_test))


# In[106]:



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[109]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[113]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,penalty='l2')
classifier.fit(x_train,y_train)


# In[115]:


y_pred=classifier.predict(x_test)


# In[117]:


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
results=pd.DataFrame([['Logistic Regression',acc,prec,rec,f1]],columns=['Model','Accuracy','Precision','recall','f1 score'])


# In[119]:


results


# In[120]:


classifier.coef_


# In[ ]:


from sklearn.svm import SVC
svm_linear_classifier=SVC(random_state=0,kernel='linear')
g=svm_linear_classifier.fit(x_train,y_train)

ypred=g.predict(x_test)
ypred.value_counts


# In[146]:


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
ypred=g.predict(x_test)
ac=accuracy_score(y_test,y_pred)
prc=precision_score(y_test,y_pred)
re=recall_score(y_test,y_pred)
f=f1_score(y_test,y_pred)
model_results=pd.DataFrame([['SVM (Linear)',ac,prc,re,f]],columns=['Model','Accuracy','Precision','recall','f1 score'])

results=results.append(model_results,ignore_index=True)


# In[147]:


model_results

