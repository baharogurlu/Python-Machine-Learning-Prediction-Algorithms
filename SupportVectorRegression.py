#Sınıflandırma algoritması olarak kullanılan destek vektör algoritması iki sınıfı ayırırken en fazla marjine(iki sınıftaki en yakın komşu aralığı) sahip olan demektir aslında
#Tahmin algoritması olarak kullanımında ise üzerinden en çok nokta geçen doğru olarak tanımlanır.Marjin değerini minimize ediyor olmalıdır.

from sklearn.preprocessing import StandardScaler
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

print("Linear Regresyon : ",lr.predict(np.array([[11]]))) #kisinin egitim seviye 11 ise maaşı linear regresyon ile hesaplandı consola yazan değer

print("Polinomial Regresyon : ",lr2.predict(pol_reg.fit_transform(np.array([[11]])))) #kisinin egitim seviye 11 ise maaşı polinomial regresyon ile hesaplandı consola yazan değer
 #sonuç olarak polinomail ile yapılan tahminin daha başarılı olduğu görülmüştür.******************************************



#SUPPORT VECTOR REGRESSION##############################################################################################################
sc=StandardScaler()
egitimSeviyesiOlcekli=sc.fit_transform(egitimSeviyesiVal)

sc2=StandardScaler()
maasOlcekli=sc2.fit_transform(maasVal)

#Değerleri standart ölçekten geçirdik

from sklearn.svm import SVR
# KERNELS RBF,LİNEAR,POLİNOMİAL
svr_reg=SVR(kernel='rbf')
svr_reg.fit(egitimSeviyesiOlcekli,maasOlcekli)

plt.scatter(egitimSeviyesiOlcekli,maasOlcekli,color="red")
plt.plot(egitimSeviyesiOlcekli,svr_reg.predict(egitimSeviyesiOlcekli),color="blue")
plt.show()

print("Support Vektor Regression RBF : ",svr_reg.predict(np.array([[11]])))

svr_lin = SVR(kernel='linear')
svr_lin.fit(egitimSeviyesiOlcekli,maasOlcekli)

plt.scatter(egitimSeviyesiOlcekli,maasOlcekli,color="black")
plt.plot(egitimSeviyesiOlcekli,svr_lin.predict(egitimSeviyesiOlcekli),color="green")
plt.show()
print("Support Vektor Regression Linear : ",svr_lin.predict(np.array([[11]])))


svr_poly = SVR(kernel='poly')
svr_poly.fit(egitimSeviyesiOlcekli,maasOlcekli)

plt.scatter(egitimSeviyesiOlcekli,maasOlcekli,color="pink")
plt.plot(egitimSeviyesiOlcekli,svr_poly.predict(egitimSeviyesiOlcekli),color="gray")
plt.show()

print("Support Vektor Regression Poly : ",svr_poly.predict(np.array([[11]])))