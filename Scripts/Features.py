#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
def features(df,on_Col,out_Col,col_Name,aggfunc='sum'):
    dummy_grp=df.groupby(on_Col)[out_Col].agg(aggfunc).to_dict()
    df[col_Name]=df[on_Col].map(dummy_grp)
    return df

