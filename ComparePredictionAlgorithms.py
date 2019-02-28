#R2 (R-square,R-Kare) YÖNTEMİ##########################################################

#R2 (R-square,R-Kare) Yöntemi : En küçük kabul edilebilir değerini üretebiliriz.
#Hata kareleri toplamı(HKT):   Toplam=Topla(yi+y'i)karesi    (hata farkları)
#Ortalama farkların(OFT) toplamı=    Topla(yi-y.ort)karesi  (tahmin verilerinin ortlaması)
#R2(R'nin karesi)=1-HKT/OFT

#Örn:boy=130,kilo=30,yas=10, tahmin=12, hata oranı=2
#Örn:boy=100,kilo=20,yas=5, tahmin=6,hata oranı=-1
#toplam=             yas=15,tahmin=18:2=9 ,hata oranı=5

#HKT1(10-12)*(10-12)=4
#HKT1(5-6)*(5-6)=1
#ToplamHKT=5

#OFT1(12-9)*(12-9)=9
#OFT1(5-9)*(5-9)=16
#ToplamOFT=25

#R2(R'nin karesi)=1-HKT/OFT
#R2(R'nin karesi)=1-5/25=0.8 en kötü 0, en iyi 1 olabilir. negatif çıkarsa çok kötüdür

#######################################################################################




#Düzeltilmiş R2 (Adjusted R-square) YÖNTEMİ##########################################################

#Düzeltilmiş R2=1-(1-R2)-((n-1)/(n-p-1))


#1. kutuphaneler
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score

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

print('POLINOMIAL LINEAR REGRESSİON: ',lr.predict(np.array([[11]]))) #kisinin egitim seviye 11 ise maaşı linear regresyon ile hesaplandı consola yazan değer


print(lr2.predict(pol_reg.fit_transform(np.array([[11]])))) #kisinin egitim seviye 11 ise maaşı polinomial regresyon ile hesaplandı consola yazan değer
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




#DECISION TREE REGRESSION##############################################################################################################
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(egitimSeviyesi,maas)
plt.scatter(egitimSeviyesi,maas,color="red")
plt.plot(egitimSeviyesi,r_dt.predict(egitimSeviyesi),color="blue")
yeniEs=egitimSeviyesi+0.5
yeniEs2=egitimSeviyesi-0.3
plt.plot(egitimSeviyesiVal,r_dt.predict(yeniEs),color="pink")
plt.plot(egitimSeviyesiVal,r_dt.predict(yeniEs2),color="yellow")
plt.show()

print("DECISION TREE REGRESSION : ",r_dt.predict(np.array([[11]])))
print("DECISION TREE REGRESSION: ",r_dt.predict(np.array([[9]])))


#RANDOM FOREST REGRESSION##############################################################################################################
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0,) #n_estimators kaç tane alt ağaca bölüneceğini belirler. random_state = 0 ifadesi, bize sonuçların tekrarlanması olanağını sağla
rf_reg.fit(egitimSeviyesi,maas)
print(rf_reg.predict(np.array([[6.5]])))

#Decision Tree veri kümesindeki sonuçlardan birini döndürürken random forest daha esnek rakamlar tahmin edebilir.
#Bunu birden fazla decison tree nin verdiği kararların ortalamasını alarak yapar.

plt.scatter(egitimSeviyesi,maas,color="red")
plt.plot(egitimSeviyesiVal,rf_reg.predict(egitimSeviyesi),color="green")

yeniEs=egitimSeviyesi+0.5
yeniEs2=egitimSeviyesi-0.6
plt.plot(egitimSeviyesiVal,rf_reg.predict(yeniEs),color="blue")
plt.plot(egitimSeviyesiVal,rf_reg.predict(yeniEs2),color="pink")
plt.show()


#R2 (R-SUARE) YÖNTEMİ#####################################################################################################################

#print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(yeniEs)))
#print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(yeniEs2)))

print('R2 SKORLARI: ')
print('LINEAR REGRESSİON R2 skoru: ',r2_score(maasVal,lr.predict(egitimSeviyesiVal)))
print('POLINOMIAL LINEAR REGRESSİON R2 skoru: ',r2_score(maasVal,lr2.predict(pol_reg.fit_transform(egitimSeviyesiVal))))
print('SUPPORT VECTOR REGRESSİON R2 skoru: ',r2_score(maasOlcekli,svr_poly.predict(egitimSeviyesiOlcekli)))# burada radyal değeri değiştirip tekrar bakmak sağlıklı olabilir
print('DECISION TREE REGRESSION R2 skoru: ',r2_score(maas,r_dt.predict(egitimSeviyesi)))#R square yüksek çıkar kanmayın
print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(egitimSeviyesi)))