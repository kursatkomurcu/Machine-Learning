#!/usr/bin/env python
# coding: utf-8

# In[133]:


from sklearn import preprocessing
import numpy as np
import pandas as pd


# In[134]:


dataset = pd.read_csv('ks-projects-201801.csv')#We read our dataset
dataset.head()


# In[135]:


dataset = dataset.dropna()#Dropping some rows that have NaN value
dataset


# In[136]:


#We reach values which is in state column
Failed = dataset[dataset.state == 'failed']
Successfull = dataset[dataset.state == 'successfull']
Canceled = dataset[dataset.state == 'canceled']
Live = dataset[dataset.state == 'live']
Suspended = dataset[dataset.state == 'suspended']


# In[137]:


#Plotting state scatter graphic according to country and goal
import matplotlib.pyplot as plt
plt.scatter(Failed.country, Failed.goal, color="red", label = "Failed", alpha = 0.3)
plt.scatter(Successfull.country, Successfull.goal, color="green", label = "Successfull", alpha = 0.3)
plt.scatter(Canceled.country, Canceled.goal, color="blue", label = "Canceled", alpha = 0.3)
plt.scatter(Live.country, Live.goal, color="yellow", label = "Live", alpha = 0.3)
plt.scatter(Suspended.country, Suspended.goal, color="black", label = "Suspended", alpha = 0.3)
plt.xlabel("country")
plt.ylabel("goal")
plt.legend()
plt.show()


# In[109]:


dataset['state'].value_counts()#Learning number of each state class


# In[110]:


#Calculating Percentage of state classes#
failed = 197611; successful = 133851; canceled = 38757; live = 2798; suspended = 1843
add = failed + successful + canceled + live + suspended
percent_failed = (failed*100) / add
percent_successful = (successful*100) / add
percent_others = ((canceled + live + suspended)*100) / add
print("Failed Percent is %{}".format(percent_failed))
print("Successful Percent is %{}".format(percent_successful))
print("Others Percent is %{}".format(percent_others))


# In[111]:


dataset.columns


# In[112]:


#We chose some values that is dependent on state. 
#Some values such as id, name are not dependent on state
x = dataset[['category', 'main_category','currency', 'goal', 'pledged', 'backers', 
            'country', 'usd pledged', 'usd_pledged_real', 'usd_goal_real']].values
x


# In[113]:


#We assign codes to values that are not number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 2] = le.fit_transform(x[:, 2])
x[:, 6] = le.fit_transform(x[:, 6])
x


# In[114]:


y = dataset[['state']].values
y


# In[115]:


#Feature scaling and fitting
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x


# In[116]:


#We split datas. 80% of these datas are training data, others are testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print("Train set: ", x_train.shape, y_train.shape)
print("Test set: ", x_test.shape, y_test.shape)


# In[117]:


#Using KNeighborsClassifier function and fitting to apply KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
k = 8
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
neigh


# In[118]:


#predicting values and printing "y prediction" and "y test" values
y_pred = neigh.predict(x_test)
print(y_pred[0:20])
print(y_test[0:10])


# In[119]:


#Printing train and test set accuracy
from sklearn import metrics
print("Train set Accuracy: %", metrics.accuracy_score(y_train, neigh.predict(x_train))*100)
print("Test set Accuracy: %", metrics.accuracy_score(y_test, y_pred)*100)


# In[131]:


#We apply knn algorithm for values that are in between 1 and 15
#Then we add score values to a list and plot K values-Accuracy Graphic
score_list = []
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(score_list)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()

