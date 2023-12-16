#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Edit this file for combining all of our codes. But Do not Edit this before you test and run on your computer.


# # New Section

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams 
rcParams['figure.figsize'] = (12,8)
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler , StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#if there are more to import, import them here.


# 

# In[3]:


#necessary import


# In[4]:


# import io
pdDf = pd.read_csv('ParisHousing2.csv')
# prices = pd.read_csv(io.BytesIO(uploaded['ParisHousing2.csv']))
prices = pdDf
print(prices)     # storing initial dataset in df


# # ========================================

# In[5]:


prices.head(8)


# In[6]:


prices.shape


# # Represent using a bar chart of N classes
# 
# 

# In[7]:


prices.hist(bins = 50, figsize = (20,15))
plt.show()

#Bins are the number of intervals you want to divide all of your data into. 
#figsize is The size of the output image


# # Pre Processsing Part starts here

# In[8]:


prices.isnull()


# In[9]:


prices.isnull().sum()


# In[10]:


print("Number of rows with null values in floors: ", prices['floors'].isnull().sum())


# In[11]:


print("Number of rows with null values in squareMeters: ", prices['squareMeters'].isnull().sum())


# In[12]:


print("Before dropping the null rows, the shape looks like this: ", prices.shape)


# In[13]:


prices = prices.dropna(axis=0, subset= ['squareMeters'])


# In[14]:


print("After dropping the null rows, the shape looks like this: ", prices.shape)


# In[15]:


impute = SimpleImputer(missing_values = np.nan, strategy='mean')
impute.fit(prices[['floors']])

prices['floors'] = impute.transform(prices[['floors']])


# In[16]:


prices.head(8)


# In[17]:


prices['isNewBuilt'].unique()


# In[18]:


# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the column
#prices['isNewBuilt'] = enc.fit_transform(prices['isNewBuilt'])

# Compare the two columns
#print(prices[['isNewBuilt']].head())

prices['isNewBuilt'] = prices['isNewBuilt'].map({'No':0, '1':1, '0':0}) 


# In[19]:


prices.head(8)


# In[19]:





# # ======= Pre Processing Part Ends here =======

# # Heatmap

# In[20]:


plt.figure(figsize=(12, 6))
sns.heatmap(prices.corr(),
            cmap = 'YlGnBu',
            fmt = '.3f',
            #linewidths = 2,
            annot = True)


# # Precision Part

# In[21]:


# using minmax scaler for scaling the entire dataset

scaler = MinMaxScaler()

# prices = prices.values

scaler.fit(prices)

data_train_scaled = scaler.transform(prices)
data_test_scaled = scaler.transform(prices)    # using same fit for both test and train


# # Splitting Part

# In[22]:


# using random dataset splitting    (NOT SURE ABOUT THIS)

from sklearn.model_selection import train_test_split

dataFeat = data_train_scaled[:,:-1]   # storing features in dataFeat
dataPrice = data_train_scaled[:,-1]   # storing label (in this case price) in dataPrice

x_train, x_test, y_train, y_test = train_test_split(dataFeat, dataPrice, test_size = 0.30, random_state=0) # splitting 70/30

print("x_train:",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

# This and EVERYTHING above are common for all .py files


# # Decision Tree Regression

# In[32]:


from sklearn.tree import DecisionTreeRegressor
# create a regressor object
regressorDT = DecisionTreeRegressor()

regressorDT.fit(x_train, y_train)

y_pred_DT = regressorDT.predict(x_test)


print("Predicted price:\n", y_pred_DT)

print("Actual price:\n",y_test)


accuracyDT = regressorDT.score(x_test,y_test) * 100

print("Accuracy of DTR:",accuracyDT)

# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


# By Precision and Recall

# In[35]:


# 7) 2nd point

# TP / (TP + FP)
# TP / (TP + FN)

from sklearn.metrics import average_precision_score, recall_score

percent_difference_y = []
for i in y_test:
  percent_difference_y.append(i)

for i in range(len(percent_difference_y)):
  percent_difference_y[i] = 1

# Above this, COMMON for all .py files




percent_difference_DT = (abs(y_test - y_pred_DT)/y_test) * 100       # for Decision Tree
for i in range(len(percent_difference_DT)):
  if percent_difference_DT[i] <= 1:
    percent_difference_DT[i] = 1
  else:
    percent_difference_DT[i] = 0
# print(percent_difference_DT)

pScoreDT = average_precision_score(percent_difference_y, percent_difference_DT)
rScoreDT = recall_score(percent_difference_y, percent_difference_DT)

print("Precision Score for Decision Tree Regression is:",pScoreDT)
print("Recall Score for Decision Tree Regression is:",rScoreDT)


# By Confusion Matrix

# In[36]:


# 7) 3rd point

# TN FP
# FN TP

from sklearn.metrics import confusion_matrix            # COMMON FOR ALL .py files


CM_DT = confusion_matrix(percent_difference_y, percent_difference_DT)       # for Decision Tree
print("Confusion Matrix for Decision Tree Regression is:\n",CM_DT,"\n")

