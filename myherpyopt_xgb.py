# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 11:08:29 2018

@author: Administrator
"""

#coding:utf-8

import pandas as pd
from sklearn.cross_validation import cross_val_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import re
from scipy import sparse
from sklearn.feature_selection import chi2, SelectPercentile
import gc
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
    
train1 = pd.read_csv('round1_iflyad_train.txt', sep='\t')
train2 = pd.read_csv('./round2/round2_iflyad_train.txt', sep='\t')
test = pd.read_csv('./round2/round2_iflyad_test_feature.txt', sep='\t')
#data=train.copy()
data_train = pd.concat([train1,train2]).reset_index(drop=True)
data_train.drop_duplicates(inplace=True)
data = pd.concat([data_train,test]).reset_index(drop=True)

###删掉单值，并列关系的特征
drop=['app_paid','creative_is_js','creative_is_voicead']
data.drop(drop,axis=1,inplace=True)

##### 填充缺失值
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))

####raplace
replace = ['creative_is_download','creative_has_deeplink','creative_is_jump']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])
    
####advert_industry_inner处理,1,2前半截是一样的，inner2取后两位
data['advert_industry_inner_1'],data['advert_industry_inner_2']=data['advert_industry_inner'].str.split('_',1).str
del data['advert_industry_inner']
data['advert_industry_inner_2']=data['advert_industry_inner_2'].apply(lambda x:int(int(x)%100) ).astype(str)

#########time
data_time_min = data['time'].min() 
data['day'] = (data['time'].values - data_time_min) / 3600 / 24
data['day'] = data['day'].astype(int)
data['hour'] = (data['time'].values - data_time_min - data['day'].values * 3600 * 24) / 3600
data['hour'] = data['hour'].astype(int) 
data['label'] = data.click.astype(int)
del data['click']
######os,osv
def find_number(x):
    res = re.findall('\d+\.?\d?\.?\d?', x)
    res = res[0] if len(res) > 0 else '-1'
    return res
data['osv_num'] = data['osv'].apply(find_number)   #操作系统版本  有-1
data['os_osv_num'] = data['os'].astype(str).values + '_' + data['osv_num'].astype(str).values

#####f_channel
f_channel = data['f_channel'].value_counts().reset_index()
f_channel=f_channel[f_channel['f_channel']==1]
data['f_channel']=data['f_channel'].apply(lambda x: "-1" if (x in f_channel['index'].values) else x).astype(str)

######adid 去掉后很差
adid = data['adid'].value_counts().reset_index()
adid=adid[adid['adid']==1]
data['adid']=data['adid'].apply(lambda x: -1 if (x in adid['index'].values) else x).astype(int)
data['adid']=data['adid'].apply(lambda x:int(x/10000))
####city
data['city']=data['city'].apply(lambda x:int(x/1000000000))
#####宽高   
data['area'] = (data['creative_height'] * data['creative_width']).astype(int)
data['w_h_ratio']=data['creative_height'] / data['creative_width']
#####编码
cate_feature =['city','adid','advert_id','advert_industry_inner_1','advert_industry_inner_2','creative_type',
'campaign_id','creative_id','orderid','app_cate_id','app_id','creative_tp_dnf','creative_has_deeplink','province',
'inner_slot_id','advert_name','f_channel','model','os_osv_num','carrier','devtype','nnt','creative_is_download','make',
'os_name','creative_is_jump']#

for i in cate_feature:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    
num_feature=['creative_height','creative_width','hour','day']

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

base_train_csr = sparse.csr_matrix((len(data_train), 0))
enc = OneHotEncoder()
for feature in cate_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr','bool')
print('one-hot prepared !')

cv = CountVectorizer(min_df=20)
for feature in ['user_tags']:
    data[feature] = data[feature].astype(str)
    cv.fit(data[feature])
    base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
print('cv prepared !')

sparse.save_npz('./feature/base_train_csr_xgb.npz', base_train_csr)

train_csr = sparse.hstack( (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype( 'float32')

print(train_csr.shape)
feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
print('feature select')
print(train_csr.shape)

def GBM(argsDict):
    max_depth=argsDict['max_depth'] +1
    n_estimators = argsDict['n_estimators'] * 50 + 50#1000 + 10
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.02
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"] + 1
    reg_lambda=argsDict["reg_lambda"]*0.1 + 0.1
    reg_alpha=argsDict["reg_alpha"]*0.1 + 0.1
    colsample_bytree=argsDict["colsample_bytree"] * 0.1 + 0.7
    seed=argsDict["seed"]
    colsample_bylevel= argsDict["colsample_bylevel"]* 0.1+0.1
    print ("max_depth"+ str(max_depth))
    print ("n_estimators:" + str(n_estimators))
    print ("learning_rate:" + str(learning_rate))
    print ("subsample:" + str(subsample))
    print ("min_child_weight:" + str(min_child_weight))
    print ("reg_lambda:" + str(reg_lambda))
    print ("reg_alpha:"+str(reg_alpha))
    print ("colsample_bytree:"+str(colsample_bytree))
    print ("seed:" +str(seed))
    print ("colsample_bylevel:" + str(colsample_bylevel))
    global train_csr,train_y

    gbm = xgb.XGBClassifier(
                             max_depth=max_depth,
                             boosting_type='gbdt',
                             nthread=8,    #进程数
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=colsample_bylevel,
                            subsample_freq=1,
                            max_delta_step = 100,  #10步不降则停止
                            n_jobs=10,
                            seed=seed,
                            objective="binary:logistic")

    metric = -cross_val_score(gbm,train_csr,train_y,cv=5,scoring="roc_auc").mean()
    #print (metric)
    return metric
space = {"max_depth":hp.randint("max_depth",8),
         "n_estimators":hp.randint("n_estimators",8),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.02,0.04
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #[0,1,2,3,4] -> [1,2,3,4,5]
         "reg_alpha":hp.randint("reg_alpha",30), 
         "reg_lambda":hp.randint("reg_lambda",50),
         "colsample_bytree":hp.randint("colsample_bytree",4),
         "seed":hp.randint("seed",3000),
         "colsample_bylevel":hp.randint("colsample_bylevel",10),
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=50)#max_evals表示想要训练的最大模型数量，越大越容易找到最优解
#print (best)
print (GBM(best))

#max_depth8
#n_estimators:400
#learning_rate:0.12000000000000001
#subsample:0.8999999999999999
#min_child_weight:2
#reg_lambda:4.2
#reg_alpha:2.7
#colsample_bytree:0.7
#seed:2993
#colsample_bylevel:1.0
#-0.770029524700621
