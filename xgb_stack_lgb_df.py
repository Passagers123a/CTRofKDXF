# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:26:59 2018

@author: Administrator
"""
from sklearn.metrics import roc_auc_score
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import log_loss
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
import os
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold
import datetime
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pandas import DataFrame

train1 = pd.read_csv('round1_iflyad_train.txt', sep='\t')
#test1 = pd.read_csv('round1_iflyad_test_feature.txt', sep='\t')
train2 = pd.read_csv('./round2/round2_iflyad_train.txt', sep='\t')
test = pd.read_csv('./round2/round2_iflyad_test_feature.txt', sep='\t')
#data=train.copy()
data_train = pd.concat([train1,train2]).reset_index(drop=True)
data_train.drop_duplicates(inplace=True)
data = pd.concat([data_train,test]).reset_index(drop=True)

###删掉单值，并列关系的特征
drop=['app_paid','creative_is_js','creative_is_voicead','os_name','creative_is_jump']
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
replace = ['creative_is_download','creative_has_deeplink']
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

#####宽高   
'''data['area'] = (data['creative_height'] * data['creative_width']).astype(int)
data['w_h_ratio']=data['creative_height'] / data['creative_width']'''
######adid 去掉后很差
adid = data['adid'].value_counts().reset_index()
adid=adid[adid['adid']==1]
data['adid']=data['adid'].apply(lambda x: -1 if (x in adid['index'].values) else x).astype(int)
data['adid']=data['adid'].apply(lambda x:int(x/10000))
####city
data['city']=data['city'].apply(lambda x:int(x/1000000000))

#####编码
cate_feature =['city','adid','advert_id','advert_industry_inner_1','advert_industry_inner_2','creative_type',
'campaign_id','creative_id','orderid','app_cate_id','app_id','creative_tp_dnf','creative_has_deeplink','province',
'inner_slot_id','advert_name','f_channel','model','os_osv_num','carrier','devtype','nnt','creative_is_download','make']#

for i in cate_feature:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    
num_feature=['creative_height','creative_width','hour','day']
predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

if os.path.exists('C:/Users/LHX/Desktop/给选手/round2/base_train_csr.npz') and False:#True
    print('load_csr---------')
    base_train_csr = sparse.load_npz('C:/Users/LHX/Desktop/给选手/round2/base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('C:/Users/LHX/Desktop/给选手/round2/base_predict_csr.npz').tocsr().astype('bool')
else:
    
    base_train_csr = sparse.csr_matrix((len(data_train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr','bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),'csr','bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr','bool')
    print('cv prepared !')

    sparse.save_npz('C:/Users/LHX/Desktop/给选手/round2/base_train_csr.npz', base_train_csr)
    sparse.save_npz('C:/Users/LHX/Desktop/给选手/round2/base_predict_csr.npz', base_predict_csr)


train_csr = sparse.hstack( (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype( 'float32')
predict_csr = sparse.hstack( (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')

print(train_csr.shape)
feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
predict_csr = feature_select.transform(predict_csr)
print('feature select')
print(train_csr.shape)
#####融合yu
kdxf=pd.read_csv('kdxf_pro_fea_10-11-18-47.csv')
yu=pd.read_csv('yu_pro_fea_baseline.csv')
df=yu.merge(kdxf,on=['instance_id'],how='left')
df['label']=data[['label']]
df['com_fea']=kdxf['pro_fea']*0.6+0.4*yu['pro_fea']
del df['instance_id']

train_fea=df[df['label']>-1]
train_fea=train_fea
predict_fea=df[df['label']==-1]
del predict_fea['label'],train_fea['label']
predict_fea=predict_fea

train_csr = sparse.hstack( (train_fea, train_csr), 'csr').astype( 'float32')
predict_csr = sparse.hstack((predict_fea, predict_csr), 'csr').astype('float32')
print(train_csr.shape)

xgb_model = xgb.XGBClassifier(max_depth=8,boosting_type='gbdt',nthread=8,n_estimators=400,learning_rate=0.12000000000000001,subsample=0.8999999999999999, 
                        min_child_weight=2,  reg_lambda=4.2,reg_alpha=2.7,colsample_bytree=0.7,
                        colsample_bylevel=1.0,subsample_freq=1, max_delta_step = 100,  n_jobs=10,
                        seed=2993,objective="binary:logistic")

skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)

for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    xgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], eval_metric='logloss',early_stopping_rounds=100)
    test_pred = xgb_model.predict_proba(predict_csr)[:, 1]
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv("xgb_lgb_baseline_%s.csv" % now, index=False)