#!/usr/bin/env python
# coding: utf-8

# In[203]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[204]:


df=pd.read_csv("C:/Users/nandhu/Downloads/Screentime-App-Details.csv")


# In[205]:


df


# In[294]:


df.head()


# In[206]:


df.tail()


# #Data Cleaning

# In[207]:


df.columns


# In[208]:


df.info()


# In[209]:


#finding if the dataset has any null values or not:


# In[210]:


print(df.isnull().sum())


# In[211]:


# descriptive statistics of the data:


# In[212]:


print(df.describe())


# In[213]:


#Data visualization :


# In[214]:


plt.plot(df['Usage'])


# In[215]:


plt.plot(df['Notifications'])


# In[216]:


plt.plot(df['Times opened'])


# In[268]:


import plotly.express as px


# In[292]:


plot= px.bar(data_frame=df, 
                x = "Date", 
                y = "Usage", 
                color="App", 
                title="Usage")
plot.show()


# In[293]:


plot= px.bar(data_frame=df, 
                x = "Date", 
                y = "Notifications", 
                color="App", 
                title="Notifications")
plot.show()


# In[ ]:





# In[220]:


plot= px.bar(data_frame=df, 
                x = "Date", 
                y = "Times opened", 
                color="App",
                title="Times Opened")
plot.show()


# #Algorithms

# In[221]:


#K-Means Algo


# In[222]:


st = df[['Usage','Notifications','Times opened']]


# In[223]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3  , n_init =10 , random_state =0 )


# In[224]:


kmeans.fit(st)
kmeans.predict(st)


# In[225]:


# using elbow method to select number of clusters


# In[226]:


## Code to find within sum of squares
wss= []
for i in range(1,11) :
    kmeans = KMeans(n_clusters = i, n_init =10 ,random_state =0 )
    kmeans.fit(st)
    wss.append(kmeans.inertia_)
    print (i, kmeans.inertia_)


# In[227]:


## Plotting the Within Sum of Squares
plt.plot(range(1,11),wss)
plt.title("The Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.show()


# In[228]:


## Vizualizing the Clusters
st['cluster']=kmeans.fit_predict(st)


# In[229]:


## Cluster Plot
plt.scatter(st.loc[st['cluster']==0,'Usage'],st.loc[st['cluster']==0,'Times opened'],s=100,c='red',label='Careful')
plt.scatter(st.loc[st['cluster']==1,'Usage'],st.loc[st['cluster']==1,'Times opened'],s=100,c='green',label='Standard')
plt.scatter(st.loc[st['cluster']==2,'Usage'],st.loc[st['cluster']==2,'Times opened'],s=100,c='blue',label='Target')
plt.scatter(st.loc[st['cluster']==3,'Usage'],st.loc[st['cluster']==3,'Times opened'],s=100,c='grey',label='Careless')
plt.scatter(st.loc[st['cluster']==4,'Usage'],st.loc[st['cluster']==4,'Times opened'],s=100,c='brown',label='Sensible')


plt.title("Results of K Means Clustering")
plt.xlabel("Usages")
plt.ylabel("Time opened ")
plt.legend()
plt.show()


# In[230]:


#Splitting data into training and testing


# In[250]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
y_test


# In[251]:


# x=df["Usage"]
# y=df["Times opened"]


# In[252]:


x_train.shape


# In[253]:


y_train.shape


# In[254]:


x_test.shape


# In[255]:


y_test.shape


# In[256]:


df.Usage.unique()


# In[257]:


#labelencoder


# In[258]:


x=df.iloc[:,0:2].values
y=df.iloc[:,2].values


# In[259]:


from sklearn.preprocessing import LabelEncoder
#for x
x_labelencoder = LabelEncoder()
x[:, 0] = x_labelencoder.fit_transform(x[:, 0])
print (x)

# for y
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)


# In[260]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
y_test


# In[261]:


y_lc =LabelEncoder()
y=y_lc.fit_transform(y)
y


# In[266]:


#Logistic regression


# In[262]:


from sklearn.linear_model import LogisticRegression
LRClassifier = LogisticRegression (random_state = 0)
LRClassifier.fit (x_train, y_train)


# In[263]:


prediction = LRClassifier.predict (x_test)


# In[264]:


from sklearn.metrics import accuracy_score


# In[265]:


print(accuracy_score(y_test, prediction))


# In[267]:


# Create confusion matrix to evaluate performance of data

from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix (y_test, prediction)

print(confusionMatrix)


# #EDA

# In[278]:


df.shape


# In[280]:


df.Usage.count()


# In[281]:


df.Usage.unique()


# In[282]:


df["Usage"].value_counts()   #count for each value


# In[283]:


df.corr()


# In[284]:


from scipy import stats


# In[287]:


s_t =stats.pearsonr(df.Usage,df.Notifications)
s_t


# In[289]:


s_t=stats.linregress(df.Usage,df.Notifications)
s_t


# In[290]:


from pandas_profiling import ProfileReport


# In[291]:


rp =ProfileReport(df)
rp


# In[ ]:




