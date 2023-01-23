#!/usr/bin/env python
# coding: utf-8

# In[2]:


# IMPORTING IMPORTANT LIBRARIES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# # IMPRTING DATASETS

# In[3]:


data=pd.read_csv('winequality-red.csv')


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[36]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


# In[37]:


data


# # SPLITTING DATA INTO TARGET AND FEATURES

# In[38]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[39]:


# train test  split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=7)


# In[40]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[41]:


# LABEL ENCODER
LE=LabelEncoder()
LE.fit(y_train)
y_train_enc=LE.transform(y_train)
LE.fit(y_test)
y_test_enc=LE.transform(y_test)


# In[42]:


y_train_enc


# # Model building and prediction

# In[43]:


model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train_enc)
y_pred=model.predict(X_test)


# In[44]:


from sklearn.metrics import accuracy_score, recall_score , precision_score , f1_score , confusion_matrix , classification_report


ACCURACY=accuracy_score(y_test_enc,y_pred)
RECALL=recall_score(y_test_enc,y_pred)
PRECISION=precision_score(y_test_enc,y_pred)
F1=f1_score(y_test_enc,y_pred)
C_M=confusion_matrix(y_test_enc,y_pred)
C_R=classification_report(y_test_enc,y_pred)


# In[45]:


print(ACCURACY)
print(RECALL)
print(F1)
print(PRECISION)
print(C_R)
print(C_M)


# In[46]:


model_1=RandomForestClassifier()
model_1.fit(X_train,y_train_enc)
y_pred_1=model_1.predict(X_test)


# In[47]:


ACCURACY_1=accuracy_score(y_test_enc,y_pred_1)
RECALL_1=recall_score(y_test_enc,y_pred_1)
PRECISION_1=precision_score(y_test_enc,y_pred_1)
F1_1=f1_score(y_test_enc,y_pred_1)
C_M_1=confusion_matrix(y_test_enc,y_pred_1)
C_R_1=classification_report(y_test_enc,y_pred_1)


# In[48]:


print(ACCURACY_1)
print(RECALL_1)
print(F1_1)
print(PRECISION_1)
print(C_R_1)
print(C_M_1)


# In[ ]:




