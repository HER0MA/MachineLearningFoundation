# -*- coding: utf-8 -*-

day_time = '_03_12_1'

import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import sys
import numpy as np
#sys.path.append('../tools')
#from tools import calculate_score,draw_feature_importance,get_result

def calculate_score(pre,real):
    if(len(pre.shape)==1):
        pre = DataFrame(pre,columns=[0])
        real = DataFrame(real,columns=[0])
    else:
        pre = DataFrame(pre,columns=[i for i in range(pre.shape[1])])
        real = DataFrame(real,columns=[i for i in range(real.shape[1])])        
        
    if(len(pre)!=len(real)):
        print ('len(pre)!=len(real)','\n')
    if(len(pre.columns)!=len(real.columns)):
        print ('len(pre.columns)!=len(real.columns)','\n')
    N = len(pre)    #N：商家总数
    T = len(pre.columns)    
    print ('N:',N,'\t','T:',T,'\n')
    
    n = 0
    t = 0
    L=0
    
    while(t<T):
        n=0
        while(n<N):
            c_it = round(pre.ix[n,t])       #c_it：预测的客流量
            c_git = round(real.ix[n,t])    #c_git：实际的客流量
            
            
            if((c_it==0 and c_git==0) or (c_it+c_git)==0 ):
                c_it=1
                c_git=1
            
            L = L+abs((float(c_it)-c_git)/(c_it+c_git))
            n=n+1
        t=t+1
    #print L
    return L/(N*T)

def get_result(result):
    if(len(result.shape)==1):
        df = DataFrame(result,columns=[0])
    else:
        df = DataFrame(result,columns=['col_'+str(i) for i in range(result.shape[1])])
    df.insert(0,'shop_id',[i for i in range(1,2001)])
    #df = pd.merge(df,df,on='shop_id')
    return df.drop('shop_id',axis=1)

train_x = pd.read_csv('train_1/train_x'+day_time+'.csv')
train_y = pd.read_csv('train_1/train_y'+day_time+'.csv')
test_x = pd.read_csv('test_1/test_x'+day_time+'.csv')
test_y = pd.read_csv('test_1/test_y'+day_time+'.csv')

x_scaled_train = preprocessing.scale(train_x)
x_scaled_test = preprocessing.scale(test_x)
#para0 = n_estimators=120,learning_rate=0.05,random_state=1,min_samples_split=2,min_samples_leaf=1,max_depth=4,max_features=140,subsample=1

#param1 = {'n_estimators':[120],'learning_rate':[0.01],'random_state':[1],'min_samples_split':[4],'min_samples_leaf':[3],'max_depth':[4],'max_features':[140],'subsample':[1]}

#param = {'subsample':[0.7,0.7,1,1,1,1,1],'min_samples_leaf':[1,1,1,1,1,1,1],'n_estimators':[200,200,200,200,200,200,100],'min_samples_split':[6,4,2,8,2,4,4],\
#        'learning_rate':[0.05,0.05,0.05,0.05,0.05,0.05,0.1],'max_features':[36,36,36,36,36,36,36],'random_state':[1,1,1,1,1,1,1]\
#        ,'max_depth':[4,4,4,4,4,4,4]}



result = DataFrame()
#result['col0'] = np.arange(1,2001)
for i in range(0,7):
    #GB = GradientBoostingRegressor(n_estimators=param['n_estimators'][i],learning_rate=0.05,random_state=1,\
    #                            min_samples_split=param['min_samples_split'][i],min_samples_leaf=1,max_depth=param['max_depth'][i],max_features=param['max_features'][i],subsample=0.85)     
    
    svr = SVR(kernel='linear', C=1.0, tol=0.01,epsilon=0.1)
    svr.fit(x_scaled_train,train_y.icol(i))
    pre = (svr.predict(x_scaled_test)).round()
    result['col'+str(i)] = pre


result = get_result(result.values)
result.to_csv('GBresult'+day_time+'.csv',index=False,header=False)

result=np.asarray(result)
result=result.astype(int)

# print result
# print test_y.values
score = calculate_score(result,test_y.values)
print (score)


#draw_feature_importance(train_x,ET)

#0: {'subsample': 1, 'learning_rate': 0.05, 'min_samples_leaf': 1, \
#'n_estimators': 200, 'min_samples_split': 4, 'random_state': 1, 'max_features': 270, 'max_depth': 4}

#1: {'subsample': 1, 'learning_rate': 0.1, 'min_samples_leaf': 3,\
# 'n_estimators': 100, 'min_samples_split': 8, 'random_state': 1, 'max_features': auto, 'max_depth': 6}

#2: {'subsample': 1, 'learning_rate': 0.05, 'min_samples_leaf': 1,\
# 'n_estimators': 200, 'min_samples_split': 2, 'random_state': 1, 'max_features': 280, 'max_depth': 4}

#3: {'subsample': 1, 'learning_rate': 0.05, 'min_samples_leaf': 1,\
# 'n_estimators': 200, 'min_samples_split': 8, 'random_state': 1, 'max_features': auto, 'max_depth': 4}

#4: {'subsample': 1, 'learning_rate': 0.05, 'min_samples_leaf': 1,\
# 'n_estimators': 200, 'min_samples_split': 2, 'random_state': 1, 'max_features': 270, 'max_depth': 4}

#5: {'subsample': 1, 'learning_rate': 0.05, 'min_samples_leaf': 1,\
# 'n_estimators': 200, 'min_samples_split': 4, 'random_state': 1, 'max_features': 280, 'max_depth': 4}

#6: {'subsample': 1, 'learning_rate': 0.10, 'min_samples_leaf': 1,\
# 'n_estimators': 100, 'min_samples_split': 4, 'random_state': 1, 'max_features': 270, 'max_depth': 4}