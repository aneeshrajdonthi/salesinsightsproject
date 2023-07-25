#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[73]:


df1 = pd.read_csv(r"C:\Users\ANEESH RAJ\OneDrive\Documents\Python Scripts\archive (2)\Bengaluru_House_Data.csv")
df1


# In[74]:


df1.shape


# In[75]:


df1["area_type"].value_counts()


# In[76]:


df2=df1.drop(["area_type","availability","balcony","society"],axis=1)
df2.head()


# In[77]:


df2.isnull().sum()


# In[78]:


df3=df2.dropna()
df3.isnull().sum()


# In[79]:


df3.shape


# In[80]:


df3['size'].unique()


# In[81]:


df3['bhk']=df3['size'].apply(lambda x : int(x.split(' ')[0]))


# In[82]:


df3.head()


# In[83]:


df3['bhk'].unique()


# In[84]:


df3.total_sqft.unique()


# In[85]:


def is_float(x):
  try :
    float(x)
  except:
      return False
  return True


# In[86]:


df3[~df3['total_sqft'].apply(is_float)]


# In[87]:


def convert_sqft_to_num(x) :
  token=x.split('-')
  if len(token)==2:
    return (float(token[0])+float(token[1]))/2
  try:
    return float(x)
  except:
    return None


# In[88]:


df4=df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[89]:


df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[90]:


len(df5.location.unique())


# In[91]:


df5.location =df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts().sort_values(ascending=False)
location_stats


# In[92]:


len(location_stats[location_stats<=10])


# In[93]:


location_stats_lessthan_10 =location_stats[location_stats<=10]
location_stats_lessthan_10


# In[94]:


len(df5.location.unique())


# In[95]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stats_lessthan_10 else x)
len(df5.location.unique())


# In[96]:


df5.head()


# In[97]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[98]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]


# In[99]:


df6['price_per_sqft'].describe()


# In[100]:


def remove_pps_outliers(df):
  df_out=pd.DataFrame()
  for key,subdf in df.groupby('location'):
    m=np.mean(subdf.price_per_sqft)
    std=np.std(subdf.price_per_sqft)
    reduced_df = subdf[(subdf.price_per_sqft>(m-std))& (subdf.price_per_sqft<=(m+std))]
    df_out=pd.concat([df_out,reduced_df],ignore_index=True)
  return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[101]:


def plot_scatter_chart(df,location):
  bhk2 = df[(df.location==location)&(df.bhk==2)]
  bhk3 = df[(df.location==location)&(df.bhk==3)]
  matplotlib.rcParams['figure.figsize']=(15,10)
  plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 bhk',s=50)
  plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 bhk',s=50)
  plt.xlabel("Total Square Feet Area")
  plt.ylabel("Price Per Square Feet")
  plt.title(location)
  plt.legend()

plot_scatter_chart(df7,"Hebbal")


# In[102]:


def remove_bhk_outliers(df):
  exclude_indices = np.array([])
  for location,location_df in df.groupby('location'):
    bhk_stats = {}
    for bhk,bhk_df in location_df.groupby('bhk'):
      bhk_stats[bhk]={
          'mean':np.mean(bhk_df.price_per_sqft),
          'std':np.std(bhk_df.price_per_sqft),
          'count':bhk_df.shape[0]
      }
    for bhk,bhk_df in location_df.groupby('bhk'):
      stats=bhk_stats.get(bhk-1)
      if stats and stats['count']>5:
        exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
  return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
df8.shape


# In[103]:


plot_scatter_chart(df8,"Hebbal")


# In[104]:


import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("price per square feet")
plt.ylabel("count")


# In[105]:


df8.bath.unique()


# In[106]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("no of bathrooms")
plt.ylabel("count")


# In[107]:


df9= df8[df8.bath<df8.bhk+2]
df9.shape


# In[108]:


df10 = df9.drop(['size','price_per_sqft'],axis=1)
df10.head()


# In[109]:


dummies=pd.get_dummies(df10.location)
dummies


# In[110]:


df11 = pd.concat([df10,dummies.drop('other',axis=1)],axis=1)
df11


# In[111]:


df12= df11.drop('location',axis=1)
df12


# In[112]:


X=df12.drop('price',axis=1)
X


# In[113]:


y=df12.price
y


# In[114]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[115]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# In[116]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

cross_val_score(LinearRegression(),X,y, cv=cv)


# In[ ]:





# In[117]:


def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    return lr_clf.predict([x])[0]


# In[119]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[121]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f :
    pickle.dump(lr_clf,f)


# In[ ]:


import JSON
columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open('columns.json','w') as f :
    f.write(json.dumps)

