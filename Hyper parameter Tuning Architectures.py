#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hyper parameter Tuning Architectures
# Maintaining accuracy rate more than 80 - 95%

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



df = pd.read_csv("datasets from NASA Promise Data Repository/csv_result-CM1.csv")
df.info()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


# pd.to_numeric(df['uniq_Op'], errors='coerce')
# df["uniq_Op"] = pd.to_numeric(df['uniq_Op'], errors='coerce').fillna(0)
# df["uniq_Opnd"] = pd.to_numeric(df['uniq_Opnd'], errors='coerce').fillna(0)
# df["total_Op"] = pd.to_numeric(df['total_Op'], errors='coerce').fillna(0)
# df["total_Opnd"] = pd.to_numeric(df['total_Opnd'], errors='coerce').fillna(0)
# df["branchCount"] = pd.to_numeric(df['branchCount'], errors='coerce').fillna(0)


X=df.drop(['id','Defective'], axis=1)
X


# import sys
# def cap_data(df):
#     for col in df.columns:
#         print("capping the ",col)
#         if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
#             percentiles = df[col].quantile([0.01,0.99]).values
#             df[col][df[col] <= percentiles[0]] = percentiles[0]
#             df[col][df[col] >= percentiles[1]] = percentiles[1]
#         else:
#             df[col]=df[col]
#     return df

# final_df=cap_data(df)

# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Defective'] = le.fit_transform(df['Defective'])
Y=df['Defective']
Y


# In[ ]:





# In[8]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=10)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
# Ravel Y to pass 1d array instead of column vector
model.fit(X_train, y_train) 
model.score(X_test,y_test)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[12]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


# In[13]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
#     'naive_bayes_multinomial': {
#         'model': MultinomialNB(),
#         'params': {}
#     },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}


# In[14]:


from sklearn.model_selection import GridSearchCV
import pandas as pd
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:




