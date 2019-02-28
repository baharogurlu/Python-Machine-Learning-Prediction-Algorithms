#1. kutuphaneler
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statsmodels.formula.api as sm

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')


yas = veriler.iloc[:,1:4].values


#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
#print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
#print(ulke)


ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
#print(ulke)

cinsiyet = veriler.iloc[:,-1:].values
#print(cinsiyet)

le = LabelEncoder()
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])
#print(cinsiyet)

ohe = OneHotEncoder(categorical_features='all')
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
#print(cinsiyet)

#kategorik verileri nümeric verilere 0-1 şeklinde dönüştürerek işlenebilir hale getiriyoruz
'''
cinsiyet2 =veriler.iloc[:,-1:].values
le=LabelEncoder()
cinsiyet2[:,0]=le.fit_transform(cinsiyet2[:,0])
ohe=OneHotEncoder(categorical_features='all')
cinsiyet2=ohe.fit_transform(cinsiyet2).toarray()
print(cinsiyet2)

'''


#numpy dizileri dataframe donusumu

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
#print(sonuc)

sonuc2 =pd.DataFrame(data = yas, index = range(22), columns = ['boy','kilo','yas'])
#print(sonuc2)


#dummy(kukla) variable ı engellemek için sadece tek kolunu aldık
sonuc3 = pd.DataFrame(data = cinsiyet[:,:1] , index=range(22), columns=['cinsiyet'])
#print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
#print(s)

s2= pd.concat([s,sonuc3],axis=1)
#print(s2)

#verilerin egitim ve test icin bolunmesi

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


#verilerle öğrenme
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#verilerle prediction yani tahmin
y_pred=regressor.predict(x_test) #y_pred, y_test ile benzer  olmalı






#KİLO TAHMİNİ#####################################################

kilo = s2.iloc[:,-3:-2].values
soltaraf=s2.iloc[:,:4]
sagtaraf=s2.iloc[:,5:]

kiloharic=pd.concat([soltaraf,sagtaraf],axis=1)

x_train, x_test,y_train,y_test = train_test_split(kiloharic,kilo,test_size=0.33, random_state=0)

regressorNew = LinearRegression()
regressorNew.fit(x_train,y_train)#eğittim

kilo_pred=regressorNew.predict(x_test);#kilo_pred, y_test ile benzer  olmalı

#BACKWARD ELIMINATION (GERİYE DOĞRU ELEME)#####################################################
print(kiloharic)
X=np.append(arr=np.ones((22,1)).astype(int),values=kiloharic,axis=1)

X_l=kiloharic.iloc[:,[0,1,2,3,4,5]].values
r=sm.OLS(endog=kilo,exog=X_l).fit()
print(r.summary()) # buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

X_l=kiloharic.iloc[:,[0,1,2,3,5]].values
r=sm.OLS(endog=kilo,exog=X_l).fit()
print(r.summary()) # buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

X_l=kiloharic.iloc[:,[0,1,2,3]].values
r=sm.OLS(endog=kilo,exog=X_l).fit()
print(r.summary()) # buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

#Eğer ileri eleme yöntemi olsaydı bütün değişkenleri tek tek deneyip en düşük p-value a sahip olanı bulurduk ###################
