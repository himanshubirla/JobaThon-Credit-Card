#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedKFold,train_test_split as tts
from sklearn.metrics import roc_auc_score
from outlier import outlier_detection,multi_outlier
from missing_value import missingValue_Treat
from scipy.stats import norm,boxcox
from imblearn.over_sampling import SMOTE

def run_gbm(clf,train,target,test,features,num_features,fit_params=None,oversample=True):
    N_Splits=10
    X_train,X_val,y_train,y_val=tts(train,target,test_size=.1,random_state=100)
    missing_features=[]
    for i in X_train.columns.values:
        a=X_train[i].isna().sum()
        if a>0:
            missingValue_Treat(df=X_train,col=i)
    for i in X_val.columns.values:
        a=X_val[i].isna().sum()
        if a>0:
            missingValue_Treat(df=X_val,col=i)


    for i in num_features:
        outlier_detection(df=X_train,col=i)
        outlier_detection(df=X_val,col=i)
    if oversample==True:
        oversample=SMOTE(sampling_strategy=.55)
        X_train,y_train=oversample.fit_resample(X_train,y_train)
        X_train,Xtest,y_train,ytest=tts(X_train,y_train,test_size=(N_Splits/100),random_state=100)
        oofs=np.zeros(len(X_train))
        pred=np.zeros(len(test))
        folds=StratifiedKFold(n_splits=N_Splits)
        for fold_,(trn_idx,vl_idx) in enumerate(folds.split(X_train,y_train)):
            print(f'------------------------------------Fold{fold_ + 1}------------------------------------')
            x_trn,y_trn=X_train[features].iloc[trn_idx],y_train.iloc[trn_idx]
            x_vl,y_vl=X_train[features].iloc[vl_idx],y_train.iloc[vl_idx]
            #Prepare Test Set
            X_test=test[features]

            #Scale Data
            num_trn=x_trn.loc[:,num_features]
            num_vl=x_vl.loc[:,num_features]
            num_val=X_val.loc[:,num_features]
            num_test=X_test.loc[:,num_features]
            numTest=Xtest.loc[:,num_features]

            scaler=Normalizer()
            _=scaler.fit(num_trn)
            num_trn=pd.DataFrame(scaler.transform(num_trn),columns=num_features)
            num_vl=pd.DataFrame(scaler.transform(num_vl),columns=num_features)
            num_val=pd.DataFrame(scaler.transform(num_val),columns=num_features)
            num_test=pd.DataFrame(scaler.transform(num_test),columns=num_features)
            numTest=pd.DataFrame(scaler.transform(numTest),columns=num_features)

            x_trn.reset_index(drop=True,inplace=True)
            x_vl.reset_index(drop=True,inplace=True)
            X_val.reset_index(drop=True,inplace=True)
            Xtest.reset_index(drop=True,inplace=True)
            y_trn.reset_index(drop=True,inplace=True)
            y_vl.reset_index(drop=True,inplace=True)
            y_val.reset_index(drop=True,inplace=True)
            ytest.reset_index(drop=True,inplace=True)
            X_test.reset_index(drop=True,inplace=True)

            x_trn=pd.concat([x_trn.drop(num_features,axis=1),num_trn],axis=1)
            x_vl=pd.concat([x_vl.drop(num_features,axis=1),num_vl],axis=1)
            X_val=pd.concat([X_val.drop(num_features,axis=1),num_val],axis=1)
            X_test=pd.concat([X_test.drop(num_features,axis=1),num_test],axis=1)
            Xtest=pd.concat([Xtest.drop(num_features,axis=1),numTest],axis=1)

            #Fit Model
            _=clf.fit(x_trn,y_trn,eval_set=[(x_trn,y_trn),(Xtest,ytest),(x_vl,y_vl)],**fit_params)
            preds_vl=clf.predict_proba(x_vl)[:,1]
            preds_test=clf.predict_proba(X_test)[:,1]
            preds_val=clf.predict_proba(X_val)[:,1]
            predstest=clf.predict_proba(Xtest)[:,1]
            roc_score=roc_auc_score(y_vl,preds_vl)
            roc_score_=roc_auc_score(y_val,preds_val)
            roc_scores_=roc_auc_score(ytest,predstest)
            
            print('ROC score for Fold {0} Validation :{1} '.format(fold_+1,roc_score))
            print('ROC score for Validation :{0} '.format(roc_score_))
            print('ROC score for Test Holdout :{0} '.format(roc_scores_))
            oofs[vl_idx]=preds_vl
            pred+=preds_test/N_Splits
        oofs_score=roc_auc_score(y_train,oofs.round())
        print('ROC for OOFs{0}'.format(oofs_score))
        return oofs,pred
    else:
        oofs=np.zeros(len(X_train))
        pred=np.zeros(len(test))
        folds=StratifiedKFold(n_splits=N_Splits)
        for fold_,(trn_idx,vl_idx) in enumerate(folds.split(X_train,y_train)):
            print(f'------------------------------------Fold{fold_ + 1}------------------------------------')
            x_trn,y_trn=X_train[features].iloc[trn_idx],y_train.iloc[trn_idx]
            x_vl,y_vl=X_train[features].iloc[vl_idx],y_train.iloc[vl_idx]
            #Prepare Test Set
            X_test=test[features]

            #Scale Data
            num_trn=x_trn.loc[:,num_features]
            num_vl=x_vl.loc[:,num_features]
            num_val=X_val.loc[:,num_features]
            num_test=X_test.loc[:,num_features]

            scaler=Normalizer()
            _=scaler.fit(num_trn)
            num_trn=pd.DataFrame(scaler.transform(num_trn),columns=num_features)
            num_vl=pd.DataFrame(scaler.transform(num_vl),columns=num_features)
            num_val=pd.DataFrame(scaler.transform(num_val),columns=num_features)
            num_test=pd.DataFrame(scaler.transform(num_test),columns=num_features)

            x_trn.reset_index(drop=True,inplace=True)
            x_vl.reset_index(drop=True,inplace=True)
            X_val.reset_index(drop=True,inplace=True)
            y_trn.reset_index(drop=True,inplace=True)
            y_vl.reset_index(drop=True,inplace=True)
            y_val.reset_index(drop=True,inplace=True)
            X_test.reset_index(drop=True,inplace=True)

            x_trn=pd.concat([x_trn.drop(num_features,axis=1),num_trn],axis=1)
            x_vl=pd.concat([x_vl.drop(num_features,axis=1),num_vl],axis=1)
            X_val=pd.concat([X_val.drop(num_features,axis=1),num_val],axis=1)
            X_test=pd.concat([X_test.drop(num_features,axis=1),num_test],axis=1)

            #Fit Model
            _=clf.fit(x_trn,y_trn,eval_set=[(x_trn,y_trn),(x_vl,y_vl)],**fit_params)
            preds_vl=clf.predict_proba(x_vl)[:,1]
            preds_test=clf.predict_proba(X_test)[:,1]
            preds_val=clf.predict_proba(X_val)[:,1]
            roc_score=roc_auc_score(y_vl,preds_vl)
            roc_score_=roc_auc_score(y_val,preds_val)
            print('ROC score for Fold {0} Validation :{1} '.format(fold_+1,roc_score))
            print('ROC score for Validation :{0} '.format(roc_score_))
            oofs[vl_idx]=preds_vl
            pred+=preds_test/N_Splits
        oofs_score=roc_auc_score(y_train,oofs.round())
        print('ROC for OOFs{0}'.format(oofs_score))
        return oofs,pred
        

