#!/usr/bin/env python
# coding: utf-8

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


cvid_url='https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnytimes%2Fcovid-19-data%2Fmaster%2Fus-counties.csv&filename=us-counties.csv'
cvid_file='us-counties.csv'

aqi_url='https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_2020.zip'
aqi_zip='daily_aqi_by_county_2020.zip'
aqi_file='daily_aqi_by_county_2020.csv'
# thresh_pcntg = 0.6 # drop parameter if fraction of null is less than this fraction


# ## Download data

# In[3]:


import requests
r=requests.get(aqi_url)
with open(aqi_zip,'wb+') as f:
    f.write(r.content) 

r=requests.get(cvid_url)
with open(cvid_file,'wb+') as f:
    f.write(r.content)

import zipfile
with zipfile.ZipFile(aqi_zip, 'r') as zip_ref:
    zip_ref.extractall('./')


# ## Read Covid-19 data for US

# In[4]:


import pandas as pd

cvid=pd.read_csv(cvid_file)
cvid.county=cvid.county.str.lower()
cvidCounties = cvid.county.unique().tolist()


# ## Read AQI data

# In[8]:


aqi=pd.read_csv(aqi_file)
aqi=aqi.rename(columns={'county Name':'City'})
aqi.City=aqi.City.str.lower()
aqiCounties=aqi.City.unique().tolist()
aqi=aqi.drop(columns=['State Code', 'County Code', 'Category', 'Number of Sites Reporting'])


# ## Consider Counties which have both Covid data and AQI info available

# In[9]:


commonCounties=set(aqiCounties).intersection(cvidCounties)
len(commonCounties)


# ## Keep rows with Counties from Common Counties only

# In[10]:


cvid=cvid[cvid.county.isin(commonCounties)]
aqi=aqi[aqi.City.isin(commonCounties)]


# ## Sync Dates (i.e. keep common dates only)

# In[11]:


min(cvid.date), max(cvid.date), min(aqi.Date), max(aqi.Date)


# In[13]:


startdate=max(min(cvid.date), min(aqi.Date))
enddate=min(max(cvid.date), max(aqi.Date))
startdate, enddate


# In[14]:


cvid=cvid[cvid.date.between(startdate, enddate, inclusive=True)]
aqi=aqi[aqi.Date.between(startdate, enddate, inclusive=True)]


# ## Split dataframe by County name
# ## Sync rows between cvid and aqi for each county
# 
# ## FillNa Method

# In[43]:


cvidByCounty={county:df for county, df in cvid.groupby('county')}
aqiByCounty={county:df for county, df in aqi.groupby('City')}

cvid_aqiByCounty={}

for county, cvidCounty in cvidByCounty.items():
    
    aqiCounty=aqiByCounty[county]
    
    aqiCountyDates=set(aqiCounty.Date)
    cvidCountyDates=set(cvidCounty.date)
    
    # filtering common dates only
    commonCountyDates=aqiCountyDates.intersection(cvidCountyDates)
    cvidByCounty[county]=cvidCounty[cvidCounty.date.isin(commonCountyDates)]
    aqiByCounty[county]=aqiCounty[aqiCounty.Date.isin(commonCountyDates)]  
    
    
    cvidByCounty[county]=cvidByCounty[county].groupby('date').agg({'cases': 'mean', 
                                            'date': lambda x: pd.unique(x)}).reset_index(drop=True)    
    
    
    aqiByCounty[county]=aqiByCounty[county].drop_duplicates(subset = 'Date', keep = 'last')
    aqiByCounty[county]=aqiByCounty[county].pivot(index='Date', columns='Defining Parameter', values='AQI')
    aqiByCounty[county].reset_index(inplace=True)

    if not len(cvidByCounty[county]):
        continue    
    
    aqiByCounty[county]=aqiByCounty[county].sort_values(by=['Date'])
    cvidByCounty[county]=cvidByCounty[county].sort_values(by=['date'])
    
    
    cvid_aqiByCounty[county]=aqiByCounty[county].copy()
    cvid_aqiByCounty[county]['cases']=cvidByCounty[county]['cases'].tolist()
    
    
#     cvid_aqiByCounty[county]=cvid_aqiByCounty[county].fillna(cvid_aqiByCounty[county].mean())
    cvid_aqiByCounty[county]=cvid_aqiByCounty[county].dropna(axis=0, how='any')
    cvid_aqiByCounty[county].reset_index(drop=True, inplace=True)
    
    if not len(cvid_aqiByCounty[county]):
        cvid_aqiByCounty.pop(county)
        continue
        
    print(county, len(cvidByCounty[county]), len(aqiByCounty[county]), len(cvid_aqiByCounty[county]))
    
    print(cvid_aqiByCounty[county].isna().sum())


# ## Concatenate all County data
# ## FillNa

# In[44]:


df=pd.concat(list(cvid_aqiByCounty.values()))
# df=df.sample(frac=0.7)
df.fillna(df.mean(), inplace=True)
df.isnull().sum()
len(df)


# ## Initialize the model

# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

model=LinearRegression()


# ## Train and Test

# In[18]:


X, y = df.drop(columns=['cases', 'Date']), df.cases
y=y.to_numpy().reshape(-1, 1)


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

poly=PolynomialFeatures(2)
scaler=MinMaxScaler()
X_train=poly.fit_transform(X_train)
X_test = poly.transform(X_test)


# In[21]:


model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred))


# In[22]:


import pickle

filename = 'regression_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[28]:


cvid_aqiByCounty['ada'].drop(columns=['Date']).corr()


# In[46]:


cvid_aqiByCounty['hanover'].corr()


# In[ ]:




