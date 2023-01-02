#!/usr/bin/env python
# coding: utf-8

# In[81]:



# CNN + Random forest architectures 
# Maintaining accuracy rate of 100%

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:


import tensorflow as tf
tf.__version__


# In[83]:


df = pd.read_csv("datasets from NASA Promise Data Repository/csv_result-CM1.csv")
df


# In[84]:


df.isna().sum()


# In[96]:


X=df.drop(['id','Defective'], axis=1)
X


# In[86]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Defective'] = le.fit_transform(df['Defective'])
Y=df['Defective']
Y


# In[87]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled


# In[289]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=10)


# In[290]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
# Ravel Y to pass 1d array instead of column vector
model.fit(X_train, y_train) 
model.score(X_test,y_test)


# In[90]:


# from sklearn.model_selection import train_test_split

# tf.random.set_seed(42)

# X_train_vaild, X_test, y_train_vaild, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

# X_train_vaild.shape, X_test.shape, y_train_vaild.shape, y_test.shape


# In[91]:


# tf.random.set_seed(42)

# X_train, X_valid, y_train, y_valid = train_test_split(X_train_vaild, y_train_vaild, test_size=0.25, random_state=42)

# X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[153]:


X_train.shape[0]


# In[168]:


import tensorflow as tf

from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau


# In[395]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(324, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(512, activation='relu'))


NN_model.add(Dense(1024, activation='relu'))


NN_model.add(Dense(1024, activation='relu'))


NN_model.add(Dense(512, activation='relu'))


NN_model.add(Dense(256, activation='relu'))



NN_model.add(Flatten())


NN_model.add(Dense(4096, activation='relu'))
NN_model.add(Dropout(0.5))

NN_model.add(Dense(4096, activation='relu'))
NN_model.add(BatchNormalization())
NN_model.add(Dropout(0.5))

X_NN = NN_model.predict(X_train)

#The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear', kernel_regularizer = tf.keras.regularizers.l1(l=0.01) ))

optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

#Compile the network :
NN_model.compile(loss=tf.keras.losses.mae,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 metrics=['MAE'])
NN_model.summary()




# In[172]:


NN_model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)


# In[ ]:





# In[174]:


results = NN_model.evaluate(X_test, y_test)
results


# In[175]:


X_NN


# In[396]:


dataset = pd.DataFrame(X_NN)


# In[397]:


dataset


# In[398]:


dataset['Defective'] = y_train
dataset


# In[399]:


dataset.info()


# In[400]:


dataset.shape


# In[401]:


X_for_RF = dataset.drop(labels = ['Defective'], axis=1)
Y_for_RF = dataset['Defective']
Y_for_RF


# In[402]:


X_for_RF.fillna(0)


# In[ ]:





# In[403]:


new_Y=Y_for_RF.astype('float32').fillna(0)


# In[404]:


# np.any(np.isnan(X_for_RF))
# np.any(np.isnan(Y_for_RF))
new_Y.dtypes


# In[419]:


X_train, X_test, y_train, y_test = train_test_split(X_for_RF, new_Y, test_size=0.20,random_state = 42)


# In[420]:


# y_t.dtypes
# new_y_t = y_t.astype('float32').fillna(0)
# new_y_t


# In[421]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
# Ravel Y to pass 1d array instead of column vector
model.fit(X_train, y_train) #For sklearn no one hot encoding
model.score(X_test,y_test)



from sklearn.metrics import mean_absolute_error as mae


# In[ ]:


error = mae(actual, calculated)


# In[423]:


y_predicted = model.predict(X_test)


# In[425]:


y_predicted






