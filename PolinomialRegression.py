#1. kutuphaneler
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar.csv')
egitimSeviyesi=veriler.iloc[:,1:2]
maas=veriler.iloc[:,2:]

egitimSeviyesiVal=egitimSeviyesi.values
maasVal=maas.values

lr=LinearRegression()
lr.fit(egitimSeviyesiVal,maasVal)
plt.scatter(egitimSeviyesiVal,maasVal,color='red')
plt.plot(egitimSeviyesiVal,lr.predict(egitimSeviyesiVal),color='blue')
plt.show()


#Polinamial Regression####################################

#2. Dereceden ####################################
pol_reg=PolynomialFeatures(degree=2) #2. dereceden polinom denklemi oluştu
egitimSeviyesiPoly=pol_reg.fit_transform(egitimSeviyesiVal)#b0+bx+bx2+bx3..... 1,2,9,16.... katsayılar oluştu

lr2=LinearRegression()
lr2.fit(egitimSeviyesiPoly,maasVal)
plt.scatter(egitimSeviyesiVal,maasVal,color='red')
plt.plot(egitimSeviyesiVal,lr2.predict(pol_reg.fit_transform(egitimSeviyesiVal)),color='green')#pol_reg.fit_transform(egitimSeviyesiVal) polinomal olacak şekilde dönüştürür
plt.show()


#6. Dereceden ####################################
pol_reg=PolynomialFeatures(degree=6) #6. dereceden polinom denklemi oluştu
egitimSeviyesiPoly=pol_reg.fit_transform(egitimSeviyesiVal)#b0+bx+bx2+bx3..... 1,2,9,16.... katsayılar oluştu


lr2=LinearRegression()
lr2.fit(egitimSeviyesiPoly,maasVal)
plt.scatter(egitimSeviyesiVal,maasVal,color='red')
plt.plot(egitimSeviyesiVal,lr2.predict(pol_reg.fit_transform(egitimSeviyesiVal)),color='green')#pol_reg.fit_transform(egitimSeviyesiVal) polinomal olacak şekilde dönüştürür
plt.show()

print(lr.predict(np.array([[11]]))) #kisinin egitim seviye 11 ise maaşı linear regresyon ile hesaplandı consola yazan değer

print(lr2.predict(pol_reg.fit_transform(np.array([[11]])))) #kisinin egitim seviye 11 ise maaşı polinomial regresyon ile hesaplandı consola yazan değer
 #sonuç olarak polinomail ile yapılan tahminin daha başarılı olduğu görülmüştür.******************************************