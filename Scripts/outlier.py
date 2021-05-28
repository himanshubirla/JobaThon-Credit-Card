#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


def outlier_detection(df,col):
    IQR=df[col].quantile(.75) - df[col].quantile(.25)
    upper_bound=df[col].quantile(.75) + (1.5*IQR)
    lower_bound=df[col].quantile(.75) - (1.5*IQR)
    df[col].clip(lower=lower_bound,upper=upper_bound,inplace=True)
    
    


# In[3]:


def multi_outlier(condition,df,on_Col,out_Col):
    sub_DF=df[df[on_Col]==condition]
    index_=sub_DF.index.values
    df.drop(index_,axis=0,inplace=True)
    outlier_detection(df=sub_DF,col=out_Col)
    data=pd.concat([df,sub_DF],axis=0)
    return data


# In[ ]:




