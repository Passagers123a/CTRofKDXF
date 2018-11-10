# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 14:59:27 2018

@author: Administrator
"""

import pandas as pd
import datetime
#0.423531
data1 = pd.read_csv('kdxf_baseline_10-11-18-47.csv')##0.424039
data2 = pd.read_csv('xgb_lgb_baseline_10-17-22-26.csv')##0.423569
data=0.5*data1['predicted_score']+0.5*data2['predicted_score']
res=data1[['instance_id']]
res['predicted_score']=data

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res.to_csv("combine_%s.csv" % now, index=False)

