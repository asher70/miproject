#customer purchase prediction
#I used :-https://www.youtube.com/watch?v=Myfvn5nGzW8
#:note this is not my code i made very few changes like changing of variables and a few changes i  used the  code available in the video.hence i am not the author of this code as i only typed from watching the above mentioned video.
#dataset:-https://drive.google.com/file/d/150ngSGpfHZsPgC2LpJPwYfLgJoCIODa0/view
from datetime import datetime,timedelta,date
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold,cross_val_score,train_test_split

minedata=pd.read_csv('OnlineRetail.csv',encoding='unicode_escape')
minedata['InvoiceDate']=pd.to_datetime(minedata['InvoiceDate']).dt.date
minedata['InvoiceDate'].describe()
minedatauk=minedata.query("Country=='United Kingdom'").reset_index(drop=True)
minedatauk
from datetime import date
minedata6m=minedatauk[(minedatauk.InvoiceDate < date(2011,9,1))&(minedatauk.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
minedatanext=minedatauk[(minedatauk.InvoiceDate >= date(2011,9,1))&(minedatauk.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
minedatanext['InvoiceDate'].describe()
minedatauser=pd.DataFrame(minedata['CustomerID'].unique())
minedatauser.columns=['CustomerID']
minedatafirstpurchase=minedata.groupby('CustomerID').InvoiceDate.min().reset_index()
minedatafirstpurchase.columns=['CustomerID','MinPurchaseDate']
minedatafirstpurchase.head()
minedatalastpurchase=minedata6m.groupby('CustomerID').InvoiceDate.max().reset_index()
minedatalastpurchase[:3]
minedatalastpurchase.columns=['CustomerID','MaxPurchaseDate']
minedatapurchasedates=pd.merge(minedatalastpurchase,minedatafirstpurchase,on='CustomerID',how='left')
minedatapurchasedates['NextPurchaseDay']=(minedatapurchasedates['MinPurchaseDate']-minedatapurchasedates['MaxPurchaseDate'])
minedatauser=pd.merge(minedatauser,minedatapurchasedates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')
minedatauser = minedatauser.fillna(method='ffill')
minedatauser['NextPurchaseDay'] = minedatauser['NextPurchaseDay'].fillna(timedelta(days=-1))
minedatauser.head()
minedatamaxpurchase=minedata6m.groupby('CustomerID').InvoiceDate.max().reset_index()
minedatamaxpurchase.columns=['CustomerID','MaxPurchasingDate']
minedatamaxpurchase['Recency']=(minedatamaxpurchase['MaxPurchasingDate'].max()-minedatamaxpurchase['MaxPurchasingDate']).dt.days
minedatauser=pd.merge(minedatauser,minedatamaxpurchase[['CustomerID','Recency']],on='CustomerID')
minedatauser.head()
minedatauser.Recency.describe()
sse={}
minedatarecency=minedatauser[['Recency']]
for k in range(1,10):
  kmeans=KMeans(n_clusters=k,max_iter=1000).fit(minedatarecency)
  minedatarecency["clusters"]=kmeans.labels_
  sse[k]=kmeans.inertia_
  plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("NUMBER OF CLUSTER")
plt.show()
kmeans=KMeans(n_clusters=4)
kmeans.fit(minedatauser[['Recency']])
minedatauser['RecencyCluster']=kmeans.predict(minedatauser[['Recency']])
def order_cluster(cluster_field_name,target_field_name,df,ascending):
  new_cluster_field_name='new_'+cluster_field_name
  df_new=df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
  df_new=df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
  df_new['index']=df_new.index
  df_final=pd.merge(df,df_new[[cluster_field_name,'index']],on=cluster_field_name)
  df_final=df_final.drop([cluster_field_name],axis=1)
  df_final=df_final.rename(columns={"index":cluster_field_name})
  return df_final
  minedatauser= order_cluster('RecencyCluster','Recency',minedatauser,False)
minedatauser.groupby('RecencyCluster')['Recency'].describe()
minedatafrequency=minedata6m.groupby('CustomerID').InvoiceDate.count().reset_index()
minedatafrequency.columns=['CustomerID','Frequency']
minedatafrequency.head()
  minedatauser=pd.merge(minedatauser,minedatafrequency,on='CustomerID')
minedatauser.head()
minedatauser.Frequency.describe()
sse={}
minedatafrequency=minedatauser[['Frequency']]
for k in range(1,10):
  kmeans=KMeans(n_clusters=k,max_iter=1000).fit(minedatafrequency)
  minedatafrequency["clusters"]=kmeans.labels_
  sse[k]=kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel('NUMBER OF CLUSTER')
plt.show()
kmeans= KMeans(n_clusters=4)
kmeans.fit(minedatauser[['Frequency']])
minedatauser['FrequencyCluster']=kmeans.predict(minedatauser[['Frequency']])
minedatauser.groupby('FrequencyCluster')['Frequency'].describe()
minedatauser=order_cluster('FrequencyCluster','Frequency',minedatauser,True)
minedata6m['Revenue']=minedata6m['UnitPrice']*minedata6m['Quantity']
minedatarevenue=minedata6m.groupby('CustomerID').Revenue.sum().reset_index()
minedatarevenue.head()
minedatauser=pd.merge(minedatauser,minedatarevenue,on='CustomerID')
minedatauser.Revenue.describe()
sse={}
minedatarevenue=minedatauser[['Revenue']]
for k in range(1,10):
  kmeans=KMeans(n_clusters=k,max_iter=1000).fit(minedatarevenue)
  minedatarevenue["clusters"]=kmeans.labels_
  sse[k]=kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("number of clusters")
plt.show()
kmeans=KMeans(n_clusters=4)
kmeans.fit(minedatauser[['Revenue']])
minedatauser['RevenueCluster']=kmeans.predict(minedatauser[['Revenue']])
minedatauser=order_cluster('RevenueCluster','Revenue',minedatauser,True)
minedatauser.groupby('RevenueCluster')['Revenue'].describe()
minedatauser.head()
minedatauser['OverallScore']=minedatauser['RecencyCluster']+minedatauser['FrequencyCluster']+minedatauser['RevenueCluster']
minedatauser.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()
minedatauser.groupby('OverallScore')['Recency'].count()
minedatauser['Segment']='Low-Value'
minedatauser.loc[minedatauser['OverallScore']>2,'Segment']='Mid-Value'
minedatauser.loc[minedatauser['OverallScore']>4,'Segment']='High-Value'
minedataorder=minedata6m[['CustomerID','InvoiceDate']]
minedataorder['InvoiceDay']=pd.to_datetime(minedata6m['InvoiceDate']).dt.date
minedataorder=minedataorder.sort_values(['CustomerID','InvoiceDate'])
minedataorder=minedataorder.drop_duplicates(subset=['CustomerID','InvoiceDate'],keep='first')
minedataorder['PrevInvoiceDate']=minedataorder.groupby('CustomerID')['InvoiceDay'].shift(1)
minedataorder['T2InvoiceDate']=minedataorder.groupby('CustomerID')['InvoiceDay'].shift(2)
minedataorder['T3InvoiceDate']=minedataorder.groupby('CustomerID')['InvoiceDay'].shift(3)
minedataorder.head()
minedataorder['DayDiff']=(minedataorder['InvoiceDate']-minedataorder['PrevInvoiceDate']).dt.days
minedataorder['DayDiff2']=(minedataorder['InvoiceDate']-minedataorder['T2InvoiceDate']).dt.days
minedataorder['DayDiff3']=(minedataorder['InvoiceDate']-minedataorder['T3InvoiceDate']).dt.days
minedatadiff=minedataorder.groupby('CustomerID').agg({'DayDiff':['mean','std']}).reset_index()
minedatadiff.columns=['CustomerID','DayDiffMean','DayDiffStd']
minedatadiff.head()
minedataorderlast=minedataorder.drop_duplicates(subset=['CustomerID'],keep='last')
minedataorderlast.head(10)
minedataorderlast=minedataorderlast.dropna()
minedataorderlast=pd.merge(minedataorderlast,minedatadiff,on='CustomerID')
minedatauser=pd.merge(minedatauser,minedataorderlast[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']])
minedatauser.head()
minedataclass=minedatauser.copy()
minedataclass=pd.get_dummies(minedataclass)
minedataclass['NextPurchaseDay'] = minedataclass['NextPurchaseDay'].astype(int)
minedataclass['NextPurchaseDayRange']=2
minedataclass.loc[minedataclass.NextPurchaseDay>=-20055200000000000,'NextPurchaseDayRange']=1
minedataclass.loc[minedataclass.NextPurchaseDay<=-23500800000000000,'NextPurchaseDayRange']=0
minedataclass.NextPurchaseDayRange.value_counts()/len(minedatauser)*100
corr=minedataclass[minedataclass.columns].corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr,annot=True,linewidths=0.2,fmt=".2f")
minedataclass=minedataclass.drop('NextPurchaseDay',axis=1)
X,y=minedataclass.drop('NextPurchaseDayRange',axis=1),minedataclass.NextPurchaseDayRange
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=44)
models=[]
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))
for name,model in models:
  kfold=KFold(n_splits=2)
  cv_result=cross_val_score(model,X_train,y_train,cv=kfold,scoring="accuracy")
  print(name,cv_result)
 xgb_model=xgb.XGBClassifier().fit(X_train,y_train)
 print('Accuracy of XGB classifier on training set{:.2f}'.format(xgb_model.score(X_train,y_train)))
 print('Accuracy of XGB classifier on test set{:.2f}'.format(xgb_model.score(X_test[X_train.columns],y_test)))
 y_pred=xgb_model.predict(X_test)
 print(classification_report(y_test,y_pred))
