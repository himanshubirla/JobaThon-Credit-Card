import numpy as np
import pandas as pd
def missingValue_Treat(df,col):
    
    if df[col].dtypes.name=='object':
        df[col].fillna(df[col].value_counts().sort_values(ascending=False).index[0],inplace=True)
    elif df[col].dtypes=='int64' or  df[col].dtypes=='int32':
        df[col].replace(np.nan, -999,inplace=True)
    elif df[col].dtypes=='float64' or  df[col].dtypes=='float32':
        df[col].replace(np.nan, -999.0,inplace=True)
    else:
        print("Error")