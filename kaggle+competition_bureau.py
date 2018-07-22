
# coding: utf-8

# In[333]:


import pandas as pd
import numpy as np
from pandas import DataFrame
import os
from pandas import Series, DataFrame
from numpy import nan as NA
import select
import string


# In[334]:


get_ipython().magic('pylab inline')


# In[335]:


#import data
df=pd.read_csv('/Users/Sherry/Desktop/kaggle competition/all/bureau.csv',sep=',')
df.head()


# In[336]:


df.shape


# In[337]:


##function for add NA flag & fill NA
def column_nan_flag(dataframe,column_name,fill_nan_number):
    if dataframe[column_name].isnull().sum()>0:
        dataframe[column_name]=dataframe[column_name].astype('float')
        dataframe[column_name+"NA_FLAG"]=0
        dataframe[column_name+"NA_FLAG"][dataframe[column_name].isnull()]=1
        dataframe[column_name]=dataframe[column_name].replace(np.nan,fill_nan_number)   
        #assert dataframe[column_name].isnull().sum()=0
    return dataframe


# In[343]:


#apply_function_columns1=df.columns.difference(['DAYS_CREDIT','DAYS_ENDDATE_FACT'])
#apply_function_columns2=['DAYS_CREDIT','DAYS_ENDDATE_FACT']


# In[167]:


df=column_nan_flag(df,'AMT_CREDIT_MAX_OVERDUE',0)


# In[168]:


df=column_nan_flag(df,'CNT_CREDIT_PROLONG',0)


# In[169]:


df['AMT_CREDIT_SUM']=df['AMT_CREDIT_SUM'].astype('float')
df['AMT_CREDIT_SUMNA_FLAG']=0
df['AMT_CREDIT_SUMNA_FLAG'][df['AMT_CREDIT_SUM'].isnull()]=1
df['AMT_CREDIT_SUM']=df['AMT_CREDIT_SUM'].replace(np.nan,0)


# In[170]:


df['AMT_CREDIT_SUM_DEBT']=df['AMT_CREDIT_SUM_DEBT'].astype('float')
df['AMT_CREDIT_SUM_DEBTNA_FLAG']=0
df['AMT_CREDIT_SUM_DEBTNA_FLAG'][df['AMT_CREDIT_SUM_DEBT'].isnull()]=1
df['AMT_CREDIT_SUM_DEBT']=df['AMT_CREDIT_SUM_DEBT'].replace(np.nan,0)


# In[171]:


df=df.drop(['AMT_CREDIT_SUM_LIMIT'],axis=1)


# In[172]:


df=column_nan_flag(df,'AMT_CREDIT_SUM_OVERDUE',0)


# In[173]:


df=df.drop(['DAYS_CREDIT_UPDATE'],axis=1)


# In[174]:


df['AMT_ANNUITY']=df['AMT_ANNUITY'].astype('float')
df['AMT_ANNUITYNA_FLAG']=0
df['AMT_ANNUITYNA_FLAG'][df['AMT_ANNUITY'].isnull()]=1
df['AMT_ANNUITY']=df['AMT_ANNUITY'].replace(np.nan,0)


# In[175]:


df=df.drop(['CREDIT_CURRENCY'],axis=1)


# In[176]:


df=column_nan_flag(df,'CREDIT_ACTIVE',0)


# In[177]:


df=column_nan_flag(df,'DAYS_CREDIT',-100000000)


# In[178]:


df=column_nan_flag(df,'CREDIT_DAY_OVERDUE',0)


# In[179]:


df['DAYS_CREDIT_ENDDATE']=df['DAYS_CREDIT_ENDDATE'].astype('float')
df['DAYS_CREDIT_ENDDATENA_FLAG']=0
df['DAYS_CREDIT_ENDDATENA_FLAG'][df['DAYS_CREDIT_ENDDATE'].isnull()]=1
df['DAYS_CREDIT_ENDDATE']=df['DAYS_CREDIT_ENDDATE'].replace(np.nan,0)


# In[180]:


df['DAYS_ENDDATE_FACT']=df['DAYS_ENDDATE_FACT'].astype('float')
df['DAYS_ENDDATE_FACTNA_FLAG']=0
df['DAYS_ENDDATE_FACTNA_FLAG'][df['DAYS_ENDDATE_FACT'].isnull()]=1
df['DAYS_ENDDATE_FACT']=df['DAYS_ENDDATE_FACT'].replace(np.nan,-1000000000)


# In[183]:


## create dummy variables
independent_variables=list(df.columns)
categorical_variables=['CREDIT_ACTIVE','CREDIT_TYPE']
df2=pd.get_dummies(df[independent_variables],columns=categorical_variables,prefix='',prefix_sep='',drop_first=True)


# In[268]:


#count number of loans for each individual borrower
count_SK_ID_BUREAU=pd.Series.to_frame(df.groupby(by='SK_ID_CURR')['SK_ID_BUREAU'].count())


# In[244]:


###group the columns by the SK-ID-CURR
df2=df.groupby(by='SK_ID_CURR').agg(['sum','mean','min','max']) #for categorical variables sum means count 


# In[245]:


#drop hierarchial index and rename the columns name 
rename_columns_name=['_'.join(col) for col in df2.columns]
df2.columns=df2.columns.droplevel(0)
df2.columns=rename_columns_name


# In[247]:


remove_columns=[col for col in df2.columns if 'NA_FLAG_max' in col or "NA_FLAG_min" in col]


# In[249]:


df2=df2.drop(remove_columns,axis=1)


# In[276]:


df2=df2.reset_index()


# In[281]:


count_SK_ID_BUREAU=count_SK_ID_BUREAU.reset_index()


# In[282]:


df2=pd.merge(left=df2,right=count_SK_ID_BUREAU,how='left',left_on=['SK_ID_CURR'],right_on=['SK_ID_CURR'])


# In[285]:


df2=df2.drop(['SK_ID_BUREAU_sum','SK_ID_BUREAU_mean','SK_ID_BUREAU_min','SK_ID_BUREAU_max'],axis=1)


# In[332]:


# fuction for check outliner of each feature
def outliner_chc(dataframe,column_numbers):
    for i in range(0,column_numbers+1):
            plt.figure(figsize=(8,6))
            quantile_table=dataframe[dataframe.columns[i]].quantile([0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1])
            x=quantile_table.index
            y=quantile_table.values
            plt.plot(x,y,'k.')       

