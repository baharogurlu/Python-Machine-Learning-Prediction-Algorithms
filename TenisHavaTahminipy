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
veriler = pd.read_csv('odev_tenis.csv')

#encoder:  Kategorik -> Numeric KISA YOL
verilerEncode= veriler.apply(LabelEncoder().fit_transform)

outlookVeri=verilerEncode.iloc[:,0:1]
ohe=OneHotEncoder(categorical_features='all')
outlookDonusum=ohe.fit_transform(outlookVeri).toarray()

humidity = veriler.iloc[:,2:3].values
#print(humidity) # tahmin edilecek değerim


outlookDF = pd.DataFrame(data = outlookDonusum, index = range(14), columns=['o','r','s'])

sonveriler=pd.concat([outlookDF,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([verilerEncode.iloc[:,3:],sonveriler],axis=1)

x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:6],sonveriler.iloc[:,6:],test_size=0.33, random_state=0)


#verilerle öğrenme

#ÇOKLU LİNEAR REGRESSION################################################
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print(y_test)
print('ÇOKLU LİNEAR REGRESSION')
print(humidityTahmin)
#print(x_test)


egitimKumesi=sonveriler.iloc[:,:6]
sonucKumesi=sonveriler.iloc[:,6:]
#BACKWARD ELIMINATION (GERİYE DOĞRU ELEME)#####################################################


X=np.append(arr=np.ones((14,1)).astype(int),values=egitimKumesi,axis=1)
X_l=egitimKumesi.iloc[:,[0,1,2,3,4,5]]
r=sm.OLS(endog=sonucKumesi,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz


egitimKumesi=egitimKumesi.iloc[:,1:]
sonucKumesi=sonucKumesi.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int),values=egitimKumesi,axis=1)
X_l=egitimKumesi.iloc[:,[0,1,2,3,4]]
r=sm.OLS(endog=sonucKumesi,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz


x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
#ÇOKLU LİNEAR REGRESSION################################################
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print('1.ADIM ÇOKLU LİNEAR REGRESSION BACKWARD ELIMINATION ')
print(humidityTahmin)

egitimKumesi=egitimKumesi.iloc[:,1:]
sonucKumesi=sonucKumesi.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int),values=egitimKumesi,axis=1)
X_l=egitimKumesi.iloc[:,[0,1,2,3]]
r=sm.OLS(endog=sonucKumesi,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
#ÇOKLU LİNEAR REGRESSION################################################
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print('2.ADIM ÇOKLU LİNEAR REGRESSION BACKWARD ELIMINATION ')
print(humidityTahmin)


egitimKumesi=egitimKumesi.iloc[:,1:]
sonucKumesi=sonucKumesi.iloc[:,1:]
X=np.append(arr=np.ones((14,1)).astype(int),values=egitimKumesi,axis=1)
X_l=egitimKumesi.iloc[:,[0,1,2]]
r=sm.OLS(endog=sonucKumesi,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliy
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
#ÇOKLU LİNEAR REGRESSION################################################
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print('3.ADIM ÇOKLU LİNEAR REGRESSION BACKWARD ELIMINATION ')
print(humidityTahmin)



egitimKumesi=pd.concat([egitimKumesi.iloc[:,0:1],egitimKumesi.iloc[:,1:2]],axis=1)
X=np.append(arr=np.ones((14,1)).astype(int),values=egitimKumesi,axis=1)
X_l=egitimKumesi.iloc[:,[0,1]]
r=sm.OLS(endog=sonucKumesi,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliy


x_train=pd.concat([x_train.iloc[:,0:1],x_train.iloc[:,1:2]],axis=1)
x_test=pd.concat([x_test.iloc[:,0:1],x_test.iloc[:,1:2]],axis=1)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print('4.ADIM ÇOKLU LİNEAR REGRESSION BACKWARD ELIMINATION ')
print(humidityTahmin)















'''
#encoder:  Kategorik -> Numeric  UZUN YOL
outlook=veriler.iloc[:,0:1].values
le =LabelEncoder()
outlook[:,0]=le.fit_transform(outlook[:,0])
#print(outlook) #0,1 .. şeklinde tek colon yaptı, daha sonra 2 den fazla değer olduğu için bunları çeşitlerine göre 0-1 şeklinde colon haline getirdi

ohe=OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()
#print(outlook)




windy=veriler.iloc[:,3:4].values
le =LabelEncoder()
windy[:,0]=le.fit_transform(windy[:,0])
#print(windy) #0,1 .. şeklinde tek colon yaptı, daha sonra 2 den fazla değer olduğu için bunları çeşitlerine göre 0-1 şeklinde colon haline getirdi

ohe=OneHotEncoder(categorical_features='all')
windy=ohe.fit_transform(windy).toarray()
#print(windy)


play=veriler.iloc[:,4:].values
le =LabelEncoder()
play[:,0]=le.fit_transform(play[:,0])
#print(play) #0,1 .. şeklinde tek colon yaptı, daha sonra 2 den fazla değer olduğu için bunları çeşitlerine göre 0-1 şeklinde colon haline getirdi

ohe=OneHotEncoder(categorical_features='all')
play=ohe.fit_transform(play).toarray()
#print(play)


temperature=veriler.iloc[:,1:2].values
#print(temperature)
#numpy dizileri dataframe donusumu

outlookSonuc = pd.DataFrame(data = outlook, index = range(14), columns=['sunny','rainy','overcast'] )
#print(outlookSonuc)

windySonuc = pd.DataFrame(data = windy, index = range(14), columns=['windy_false','windy_true'] )
#print(windySonuc)

playSonuc = pd.DataFrame(data = play, index = range(14), columns=['play_yes','play_no'] )
#print(playSonuc)

#dummy(kukla) variable ı engellemek için sadece tek kolunu aldık
humiditySonuc = pd.DataFrame(data = humidity[:,:1] , index=range(14), columns=['humidity'])
#print(humiditySonuc)

temperatureSonuc = pd.DataFrame(data = temperature, index = range(14), columns=['temperature'] )
#print(temperatureSonuc)




outlookWindy=pd.concat([outlookSonuc,windySonuc],axis=1)
outlookWindyPlay=pd.concat([outlookWindy,playSonuc],axis=1)
outlookWindyPlayTemp=pd.concat([temperatureSonuc,outlookWindyPlay],axis=1)

#tumVeri=pd.concat([outlookWindyPlayTemp,humiditySonuc],axis=1) #verinin ilk halini işledik ve kullanılır hali

print(outlookWindy)

x_train, x_test,y_train,y_test = train_test_split(outlookWindyPlayTemp,humiditySonuc,test_size=0.33, random_state=0)


#verilerle öğrenme

#ÇOKLU LİNEAR REGRESSION################################################
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print(y_test)
print(' ÇOKLU LİNEAR REGRESSION')
print(humidityTahmin)
#print(x_test)


#BACKWARD ELIMINATION (GERİYE DOĞRU ELEME)#####################################################
X=np.append(arr=np.ones((14,1)).astype(int),values=outlookWindyPlayTemp,axis=1)
X_l=outlookWindyPlayTemp.iloc[:,[0,1,2,3,4,5,6,7]]

r=sm.OLS(endog=humiditySonuc,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz


X_l=outlookWindyPlayTemp.iloc[:,[0,1,2,4,5,6,7]]

r=sm.OLS(endog=humiditySonuc,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz


X_l=outlookWindyPlayTemp.iloc[:,[0,2,4,5,6,7]]

r=sm.OLS(endog=humiditySonuc,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz


X_l=outlookWindyPlayTemp.iloc[:,[0,4,5,6,7]]

r=sm.OLS(endog=humiditySonuc,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

X_l=outlookWindyPlayTemp.iloc[:,[4,5,6,7]]

r=sm.OLS(endog=humiditySonuc,exog=X_l).fit()
#print(r.summary())# buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

#BACKWARD ELIMINATION (GERİYE DOĞRU ELEME)'den sonra elimizde kalan kolonlar ile tekrar tahmin işlemi yaptık ve sonucun iyileştiğini gördük#####################################################
x_train, x_test,y_train,y_test = train_test_split(X_l,humiditySonuc,test_size=0.33, random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
#verilerle prediction yani tahmin
humidityTahmin=regressor.predict(x_test)#y_pred, y_test ile benzer  olmalı
print('BACKWARD ELIMINATION (GERİYE DOĞRU ELEME) den sonra ÇOKLU LİNEAR REGRESSION')
print(humidityTahmin)
#print(x_test)

'''
