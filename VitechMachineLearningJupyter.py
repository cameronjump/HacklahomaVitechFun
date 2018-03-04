
# coding: utf-8

# In[447]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

cmap_light = ListedColormap(['#cecece', '#b2f697','#a6acfb'])
cmap_bold = ListedColormap(['#7e7e7e', '#4adf0f','#4250ff'])

import urllib.request, json 
from datetime import datetime
from datetime import timedelta


# In[ ]:




# In[65]:

id_start = 1310

'xxx'+str(id_start)


# In[54]:

data={}
data[0]=1
data


# In[208]:

id_start = str(264001)
id_end = str(265001)
urls={}
urls[0]='https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=1000'
urls[1]='https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=1000'
urls[2]='https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=1000'
urls[3]='https://v3v10.vitechinc.com/solr/v_us_plan_detail/select?indent=on&q=*&wt=json&rows=1000'

data={}
df={}
for i  in range(0,4):
    with urllib.request.urlopen(urls[i]) as url:
        data = json.loads(url.read().decode())
        to_df = data['response']['docs']
        df[i]=pd.DataFrame(to_df)

df1=pd.merge(df[0],df[1],on='id')
df_final=pd.merge(df1,df[2],on='id')
df_plans = df[3]
df_final.head()


# In[ ]:




# In[ ]:

###


# In[300]:

frames={}
df_final={}
for i in range(0,240):
    i=np.random.randint(1,1482000)
    id_start = str(i)
    id_end = str(i+10)
    urls={}
    urls[0]='https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=2000000'
    urls[1]='https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=2000000'
    urls[2]='https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&q=id:['+id_start+'%20TO%20'+id_end+']&wt=json&rows=2000000'
    
    data={}
    df={}
    for j  in range(0,3):
        with urllib.request.urlopen(urls[j]) as url:
            data = json.loads(url.read().decode())
            to_df = data['response']['docs']
            df[j]=pd.DataFrame(to_df)

    df1=pd.merge(df[0],df[1],on='id')
    df_final[i]=pd.merge(df1,df[2],on='id')
    if i%10==0:
        print(i)
frames=pd.concat(df_final)


# In[303]:

#frames.to_excel('frames.xlsx')


# In[315]:

frames.columns


# In[374]:

#frames['TOBACCO']


# In[448]:

def dip1(x):
    if x=='Yes':
        dippers=1
    else:
        dippers=0
    return dippers


# In[449]:

def income1(x):
    return x/100000


# In[450]:

frames1=pd.read_excel('frames.xlsx')


# In[451]:

frames1.head()


# In[452]:

ml=pd.DataFrame()
ml['age']=frames1['DOB'].apply(age_calc)
#ml['income']=frames['ANNUAL_INCOME']
ml['income']=frames1['ANNUAL_INCOME'].apply(income1)
ml['ppl']=frames1['PEOPLE_COVERED']
ml['plt']=frames1['PLATINUM']
ml['dip']=frames1['TOBACCO'].apply(dip1)
ml_train=ml.drop('plt',axis=1)


# In[ ]:

###REGRESSION###


# In[724]:

frames1.columns


# In[798]:

ml2=pd.DataFrame()
ml2['age']=frames1['DOB'].apply(age_calc)
ml2['income']=frames1['ANNUAL_INCOME'].apply(income1)
ml2['ppl']=frames1['PEOPLE_COVERED']
ml2['married']=frames1['MARITAL_STATUS'].apply(wedding)
ml2['purch']=frames1['PURCHASED']

ml2['Gold']=frames1['GOLD']
ml2['Silver']=frames1['SILVER']
ml2['Bronze']=frames1['BRONZE']
ml2['Platinum']=frames1['PLATINUM']

ml2['p_gold']=ml2['purch']=='Gold'
ml2['p_silver']=ml2['purch']=='Silver'
ml2['p_bronze']=ml2['purch']=='Bronze'
ml2['p_plt']=ml2['purch']=='Platinum'

ml2['pp']=ml2['Gold']*ml2['p_gold']
ml2['pp']=ml2['Silver']*ml2['p_silver']+ml2['pp']
ml2['pp']=ml2['Bronze']*ml2['p_bronze']+ml2['pp']
ml2['pp']=ml2['Platinum']*ml2['p_plt']+ml2['pp']

ml2['dip']=frames1['TOBACCO'].apply(dip1)
ml2['job']=frames1['EMPLOYMENT_STATUS'].apply(job1)
ml2['sex']=frames1['sex'].apply(sexy)


ml2['low']=frames1['PRE_CONDITIONS'].apply(low1)
ml2['med']=frames1['PRE_CONDITIONS'].apply(med1)
ml2['high']=frames1['PRE_CONDITIONS'].apply(four20)

ml2=ml2.drop('p_gold',axis=1)
ml2=ml2.drop('p_silver',axis=1)
ml2=ml2.drop('p_bronze',axis=1)
ml2=ml2.drop('p_plt',axis=1)
ml2=ml2.drop('purch',axis=1)

#ml2_train=ml2.drop('purch',axis=1)
ml2_train=ml2.drop('pp',axis=1)


# In[801]:

ml2.head()


# In[799]:

ml2_train.head()


# In[731]:

ml2.head()


# In[ ]:




# In[802]:

###PREDICTING PP###
algo,name=linear_model.BayesianRidge(),'Bayes'
#algo,name=SVR(kernel='linear', C=1e3),'SVR'
#algo,name=SVR(kernel='rbf', C=1e3, gamma=0.1),'SVR'
#algo,name=GaussianProcessRegressor(),'GP'

x_train, x_test, y_train, y_test = train_test_split(ml2_train, ml2['pp'], test_size=.1, random_state=1)

algo.fit(x_train,y_train)
pred=algo.predict(x_test)

c=pd.DataFrame(y_test)
c['pred']=pred

print(algo.score(ml2_train,ml2['pp']))
print(mean_absolute_error(y_test,pred))


# In[803]:

c.head(20)


# In[804]:

sns.lmplot('pp','pred',c,fit_reg=False)
plt.savefig('reg.png',dpi=100)


# In[786]:

ml2=pd.DataFrame()
ml2['age']=frames1['DOB'].apply(age_calc)
ml2['income']=frames1['ANNUAL_INCOME'].apply(income1)
ml2['ppl']=frames1['PEOPLE_COVERED']
ml2['married']=frames1['MARITAL_STATUS'].apply(wedding)
ml2['purch']=frames1['PURCHASED']

ml2['Gold']=frames1['GOLD']
#ml2['Silver']=frames1['SILVER']
#ml2['Bronze']=frames1['BRONZE']
ml2['Platinum']=frames1['PLATINUM']

#ml2['p_gold']=ml2['purch']=='Gold'
#ml2['p_silver']=ml2['purch']=='Silver'
#ml2['p_bronze']=ml2['purch']=='Bronze'
#ml2['p_plt']=ml2['purch']=='Platinum'

#ml2['pp']=ml2['Gold']*ml2['p_gold']
#ml2['pp']=ml2['Silver']*ml2['p_silver']+ml2['pp']
#ml2['pp']=ml2['Bronze']*ml2['p_bronze']+ml2['pp']
#ml2['pp']=ml2['Platinum']*ml2['p_plt']+ml2['pp']

ml2['dip']=frames1['TOBACCO'].apply(dip1)
ml2['job']=frames1['EMPLOYMENT_STATUS'].apply(job1)
ml2['sex']=frames1['sex'].apply(sexy)


ml2['low']=frames1['PRE_CONDITIONS'].apply(low1)
ml2['med']=frames1['PRE_CONDITIONS'].apply(med1)
ml2['high']=frames1['PRE_CONDITIONS'].apply(four20)

#ml2=ml2.drop('p_gold',axis=1)
#ml2=ml2.drop('p_silver',axis=1)
#ml2=ml2.drop('p_bronze',axis=1)
#ml2=ml2.drop('p_plt',axis=1)
ml2=ml2.drop('purch',axis=1)

#ml2_train=ml2.drop('purch',axis=1)
#ml2_train=ml2.drop('pp',axis=1)
ml2_train=ml2.drop('Gold',axis=1)


# In[ ]:




# In[782]:

comp=pd.DataFrame()

comp['Gold']=frames1['GOLD']
comp['Silver']=frames1['SILVER']
comp['Bronze']=frames1['BRONZE']
comp['Platinum']=frames1['PLATINUM']


# In[785]:

comp[comp['Gold']==85.65]


# In[781]:




# In[ ]:




# In[787]:

algo,name=linear_model.BayesianRidge(),'Bayes'
#algo,name=SVR(kernel='linear', C=1e3),'SVR'
#algo,name=SVR(kernel='rbf', C=1e3, gamma=0.1),'SVR'
#algo,name=GaussianProcessRegressor(),'GP'

x_train, x_test, y_train, y_test = train_test_split(ml2_train, ml2['Gold'], test_size=.1, random_state=11)

algo.fit(x_train,y_train)
pred=algo.predict(x_test)

c=pd.DataFrame(y_test)
c['pred']=pred

print(algo.score(ml2_train,ml2['Gold']))
print(mean_absolute_error(y_test,pred))


# In[ ]:




# In[788]:

sns.lmplot('Gold','pred',c,fit_reg=False)
plt.xlabel('Gold Price (actual)',fontsize=15)
plt.ylabel('Gold Price (predicted)',fontsize=15)
#plt.axis([70, 70, 270, 270])
plt.xlim(50, 275)
#plt.xticks(25)
plt.ylim(50, 275)
#plt.figure(figsize=(7,8),facecolor='w',frameon=True,edgecolor='w')
#plt.savefig('gold svm.png',dpi=140)


# In[657]:

plan = {'Bronze','Silver','Gold','Platinum'}

ann_pay = np.zeros(ml2.shape[0])
x = np.zeros(ml2.shape[0])
poai = np.zeros(ml2.shape[0])
i = 0
for index, row in ml2.iterrows():
    for pl in plan:
        if row['purch'] == pl:
            x=row[pl]
            #print(i)
            #ann_pay[i] = 12 * row[pl[0]]
            #poai[i] = ann_pay[i]/row['income']
    i += 1
ml2['x'] = x   
#ml2['ann_pay'] = ann_pay
#ml2['poai'] = poai
#ml2['pp'] = 
ml2.head()


# In[689]:

ml2.head(10)


# In[679]:

ml2['p_gold']=ml2['purch']=='Gold'
ml2['p_silver']=ml2['purch']=='Silver'
ml2['p_bronze']=ml2['purch']=='Bronze'
ml2['p_plt']=ml2['purch']=='Platinum'


# In[674]:

def tr(x):
    if x==True:
        y=1
    else:
        y=0
    return y


# In[680]:

ml2['boog']=ml2['p_gold'].apply(tr)
ml2['boos']=ml2['p_silver'].apply(tr)
ml2['boob']=ml2['p_bronze'].apply(tr)
ml2['boop']=ml2['p_plt'].apply(tr)


# In[688]:

ml2['pp']=ml2['Gold']*ml2['p_gold']
ml2['pp']=ml2['Silver']*ml2['p_silver']+ml2['pp']
ml2['pp']=ml2['Bronze']*ml2['p_bronze']+ml2['pp']
ml2['pp']=ml2['Platinum']*ml2['p_plt']+ml2['pp']


# In[652]:

x


# In[683]:

ml2['Silver']*ml2['p_silver']


# In[ ]:




# In[591]:

def to_int(x):
    if x=='Bronze':
        y=1  
    elif x=='Silver':
        y=2  
    elif x=='Gold':
        y=3
    elif x=='Platinum':
        y=4
    else:
        y=0
    return y

def wedding(x):
    if x=='S':
        y=0  
    elif x=='M':
        y=1
    else:
        y=0
    return y

def sexy(x):
    if x=='M':
        y=1 
    elif x=='F':
        y=0
    else:
        y=0
    return y

def job1(x):
    if x=='Employed':
        y=1 
    elif x=='Unemployed':
        y=0
    else:
        y=0
    return y

def low1(x):
    if type(x)!=str:
        y=0
    else:
        y=x.count('Low')
    return y

def med1(x):
    if type(x)!=str:
        y=0
    else:
        y=x.count('Med')
    return y

def four20(x):
    if type(x)!=str:
        y=0
    else:
        y=x.count('High')
    return y


# In[ ]:

###CLASSIFICATION###


# In[789]:

ml2=pd.DataFrame()
ml2['age']=frames1['DOB'].apply(age_calc)
ml2['income']=frames1['ANNUAL_INCOME'].apply(income1)
ml2['ppl']=frames1['PEOPLE_COVERED']
ml2['married']=frames1['MARITAL_STATUS'].apply(wedding)
ml2['gold']=frames1['GOLD']
ml2['dip']=frames1['TOBACCO'].apply(dip1)
ml2['job']=frames1['EMPLOYMENT_STATUS'].apply(job1)
ml2['sex']=frames1['sex'].apply(sexy)
ml2['purch']=frames1['PURCHASED']

ml2['low']=frames1['PRE_CONDITIONS'].apply(low1)
ml2['med']=frames1['PRE_CONDITIONS'].apply(med1)
ml2['high']=frames1['PRE_CONDITIONS'].apply(four20)

#ml2['purch']=frames1['PURCHASED'].apply(to_int)
ml2_train=ml2.drop('purch',axis=1)


# In[ ]:




# In[ ]:




# In[792]:

ml2.head()


# In[795]:

#algo,name=linear_model.BayesianRidge(),'Bayes'
#algo,name=SVR(kernel='linear', C=1e3),'SVR'
#algo,name=SVR(kernel='rbf', C=1e3, gamma=0.1),'SVR'
#algo,name=GaussianProcessRegressor(),'GP'
#algo,name=GaussianProcessClassifier(),'GP'
algo,name=DecisionTreeClassifier(),'Tree'

x_train, x_test, y_train, y_test = train_test_split(ml2_train, ml2['purch'], test_size=.1, random_state=1)

algo.fit(x_train,y_train)
pred=algo.predict(x_test)

c=pd.DataFrame(y_test)
c['pred']=pred

#print(algo.score(ml_train,ml['plt']))
#print(mean_absolute_error(y_test,pred))


# In[797]:

c


# In[ ]:




# In[796]:

algo.score(x_test,y_test)


# In[430]:

age=40 #X
income=9 #Z1
ppl=1 #Z2
gold=200 #Y
dip=1 #Z3

pred1=algo.predict([age,income,ppl,gold,dip])[0]
#print("%.2f" % CO2_out[0])
pred1


# In[460]:

fig, ax = plt.subplots( nrows=1, ncols=1 )
h=2
x_min,x_max=20,70
y_min,y_max=50,150

xx,z1,z2,yy,z3 = np.meshgrid(np.arange(x_min, x_max, h),np.ones(1)*10,np.ones(1)*10,np.arange(y_min, y_max, h),np.ones(1)*10)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
grid=np.c_[xx.ravel(),z1.ravel(), z2.ravel(),yy.ravel(),z3.ravel()]

Z = algo.predict(grid)
Z = Z.reshape(xx.shape)

#fig = plt.figure(figsize=(10,8),facecolor='white')
#ax = fig.gca(projection='3d')
#plt.style.use('seaborn-whitegrid')

#surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm,linewidth=1, antialiased=False,)
cmap_light = ListedColormap(['#cecece', '#b2f697','#a6acfb','#a6acfb'])

ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

ax.set_zlim(Z.min(), Z.max())
ax.set_ylim(y_min,y_max)
ax.set_xlim(x_min,x_max)



# In[439]:

fig, ax = plt.subplots( nrows=1, ncols=1 )
h=2
x_min,x_max=20,70
y_min,y_max=50,150

xx,z1,z2,yy,z3 = np.meshgrid(np.arange(x_min, x_max, h),np.ones(1)*10,np.ones(1)*10,np.arange(y_min, y_max, h),np.ones(1)*10)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
grid=np.c_[xx.ravel(),z1.ravel(), z2.ravel(),yy.ravel(),z3.ravel()]

Z = algo.predict(grid)
Z = Z.reshape(xx.shape)

#fig = plt.figure(figsize=(10,8),facecolor='white')
#ax = fig.gca(projection='3d')
#plt.style.use('seaborn-whitegrid')

#surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm,linewidth=1, antialiased=False,)
cmap_light = ListedColormap(['#cecece', '#b2f697','#a6acfb','#a6acfb'])

ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

ax.set_zlim(Z.min(), Z.max())
ax.set_ylim(y_min,y_max)
ax.set_xlim(x_min,x_max)



# In[ ]:




# In[419]:

from sklearn.tree import DecisionTreeClassifier


# In[463]:

train2 = ml2_train.drop('income',axis=1)
train2 = train2.drop('ppl',axis=1)
train2 = train2.drop('dip',axis=1)


# In[ ]:




# In[475]:

Z


# In[515]:

fig, ax = plt.subplots( nrows=1, ncols=1 )
h=2
x_min,x_max=20,70
y_min,y_max=50,150

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
grid=np.c_[xx.ravel(),yy.ravel()]

Z = algo.predict(grid)
Z = Z.reshape(xx.shape)

#fig = plt.figure(figsize=(10,8),facecolor='white')
#ax = fig.gca(projection='3d')
#plt.style.use('seaborn-whitegrid')

#surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm,linewidth=1, antialiased=False,)
cmap_light = ListedColormap(['#cecece', '#b2f697','#a6acfb','#a5acfb'])
cmap_bold = ListedColormap(['#7e7e7e', '#4adf0f','#4250ff','#a5acfb'])

ax.pcolormesh(xx, yy, Z,cmap=cmap_bold)

#ax.set_zlim(Z.min(), Z.max())
ax.set_ylim(y_min,y_max)
ax.set_xlim(x_min,x_max)



# In[496]:

ml2=pd.DataFrame()
ml2['age']=frames1['DOB'].apply(age_calc)
ml2['income']=frames1['ANNUAL_INCOME'].apply(income1)
#ml2['ppl']=frames1['PEOPLE_COVERED']
#ml2['gold']=frames1['GOLD']
#ml2['dip']=frames1['TOBACCO'].apply(dip1)
#ml2['purch']=frames1['PURCHASED']
ml2['purch']=frames1['PURCHASED'].apply(to_int)
ml2_train=ml2.drop('purch',axis=1)


# In[ ]:




# In[513]:

#algo,name = LinearSVC(C=1),'Linear SVC'

#algo,name=linear_model.BayesianRidge(),'Bayes'
#algo,name=SVR(kernel='linear', C=1e3),'SVR'
#algo,name=SVR(kernel='rbf', C=1e3, gamma=0.1),'SVR'
#algo,name=GaussianProcessRegressor(),'GP'
algo,name=GaussianProcessClassifier(),'GP'
#algo,name=DecisionTreeClassifier(),'Tree'

x_train, x_test, y_train, y_test = train_test_split(train2, ml2['purch'], test_size=.4, random_state=11)

algo.fit(x_train,y_train)
pred=algo.predict(x_test)

c=pd.DataFrame(y_test)
c['pred']=pred

#print(algo.score(ml_train,ml['plt']))
#print(mean_absolute_error(y_test,pred))


# In[514]:

c


# In[422]:

algo.score(x_test,y_test)


# In[ ]:




# In[396]:

frames1['BRONZE'].nunique()


# In[395]:

frames1['SILVER'].nunique()


# In[394]:

frames1['GOLD'].nunique()


# In[393]:

frames1['PLATINUM'].nunique()


# In[ ]:

###PRICES EXTRAPOLATION###


# In[397]:

comp=pd.DataFrame()
comp['bronze']=frames1['BRONZE']
comp['silver']=frames1['SILVER']
comp['gold']=frames1['GOLD']
comp['plt']=frames1['PLATINUM']


# In[403]:

comp.head(20)


# In[404]:

comp[comp['bronze']==23.65]


# In[402]:

comp[comp['bronze']==25.65]


# In[399]:

comp.head(20)


# In[ ]:




# In[398]:

sns.lmplot('bronze','gold',comp)


# In[ ]:




# In[ ]:




# In[ ]:




# In[371]:

sns.lmplot('income','gold',ml)


# In[369]:

age=30
income=9
ppl=1
dip=1

pred1=algo.predict([age,income,ppl,dip])[0]
#print("%.2f" % CO2_out[0])
pred1


# In[ ]:

age	income	ppl	gold	dip


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[304]:

#np.random.randint(1,1482000)


# In[209]:

df_final['bronze_over']=df_final['BRONZE']-20
df_final['silver_over']=df_final['SILVER']-40
df_final['gold_over']=df_final['GOLD']-70
df_final['plt_over']=df_final['PLATINUM']-110


# In[305]:

#df_final['PRE_CONDITIONS']


# In[277]:

df_final.columns


# In[210]:

df_final['age']=df_final['DOB'].apply(age_calc)


# In[ ]:




# In[211]:

df_final.head()


# In[212]:

df_final.shape


# In[213]:

comp=pd.DataFrame()


# In[214]:

comp['bronze']=df_final['bronze_over']
comp['silver']=df_final['silver_over']
comp['gold']=df_final['gold_over']
comp['plt']=df_final['plt_over']


# In[215]:

comp.head()


# In[230]:

comp=pd.DataFrame()
comp['bronze']=df_final['bronze_over']
comp['silver']=df_final['silver_over']
comp['gold']=df_final['gold_over']
comp['plt']=df_final['plt_over']


# In[240]:

ml = pd.DataFrame()
ml['gold']=df_final['gold_over']
ml['age']=df_final['age']
ml['height']=df_final['HEIGHT']
ml['weight']=df_final['WEIGHT']

ml_train = ml.drop('gold',axis=1)


# In[237]:

###MACHINE LEARNING###


# In[256]:

algo,name=linear_model.BayesianRidge(),'Bayes'
#algo,name=SVR(kernel='linear', C=1e3),'SVR'
#algo,name=SVR(kernel='rbf', C=1e3, gamma=0.1),'SVR'
#algo,name=GaussianProcessRegressor(),'GP'

x_train, x_test, y_train, y_test = train_test_split(ml_train, ml['gold'], test_size=.4, random_state=11)

algo.fit(x_train,y_train)
pred=algo.predict(x_test)

c=pd.DataFrame(y_test)
c['pred']=pred

print(algo.score(ml_train,ml['gold']))
print(mean_absolute_error(y_test,pred))


# In[257]:

sns.lmplot('gold','pred',c)


# In[252]:

age = 30
height = 20
weight = 180

gold_pred = algo.predict([age,height,weight])[0]
gold_pred


# In[ ]:




# In[247]:

x_train


# In[246]:

algo.predict(20,72,185)


# In[245]:

ml_train.head()


# In[217]:

df_final.columns


# In[218]:

df_plans


# In[258]:

df_final['DOB'][0]


# In[272]:

df_final.apply(h)


# In[269]:

new=pd.DataFrame()
#new['age']=df_final.apply(age_calc1)


# In[260]:

new.head()


# In[219]:

###FUNCTIONS###


# In[220]:

def age_calc(x):
    years = 2018-int(x[:4])
    months = 3-int(x[5:7])
    age = years + months/12
    return age


# In[263]:

def age_calc1(y):
    x=y['DOB']
    years = 2018-int(x[:4])
    months = 3-int(x[5:7])
    age = years + months/12
    return age


# In[271]:

def h(y):
    y['DOB']
    return


# In[ ]:




# In[ ]:

###MISC###


# In[ ]:




# In[132]:

di = df_final['PRE_CONDITIONS'][0]


# In[133]:

di


# In[112]:

df_final['PRE_CONDITIONS'][8]


# In[78]:

sns.lmplot('longitude','latitude',df_final)


# In[82]:

df_plans


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[39]:

import urllib.request, json 
import pandas as pd

#my_url = 'https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=sex:F&state:Alaska&latitude=55&wt=json&rows=100'
my_url = 'https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'
my_url2 = 'https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'
with urllib.request.urlopen(my_url) as url:
    data = json.loads(url.read().decode())
    
with urllib.request.urlopen(my_url2) as url:
    data = json.loads(url.read().decode())

my_url3 = 'https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'

to_df2 = data['response']['docs']

urls = []
urls[0] = 'https://v3v10.vitechinc.com/solr/v_us_participant/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'
urls[1] = 'https://v3v10.vitechinc.com/solr/v_us_participant_detail/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'
ulrs[2] = 'https://v3v10.vitechinc.com/solr/v_us_quotes/select?indent=on&q=id:[1310%20TO%201320]&wt=json&rows=100'

for i  in range(0,3):
    print(i)

df1 = pd.DataFrame(to_df)
df2 = pd.DataFrame(to_df2)
df=pd.merge(df1,df2,on='id')
df.head()


# In[ ]:




# In[ ]:




# In[33]:



