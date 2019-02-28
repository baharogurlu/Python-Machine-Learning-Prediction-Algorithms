import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.formula.api as sm

veriler = pd.read_csv('maaslar_yeni.csv')
ukp=veriler.iloc[:,2:5]
maas=veriler.iloc[:,5:]

#BACKWARD ELIMINATION burada işe yarar colonları bulmaya çalışıyoruz
ukpVal=ukp.values
maasVal=maas.values


print(veriler.corr())
#BACKWARD ELIMINATION ######################################

#r=sm.OLS(endog=maas,exog=ukp.iloc[:,[0,1,2]].values).fit()
#print(r.summary()) # buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

#r=sm.OLS(endog=maas,exog=ukp.iloc[:,[0,1]].values).fit()
#print(r.summary()) # buradan    P>|t| colonundaki en büyük değerre sahip olan p-value u eliyoruz

#LINEAR REGRESSION ######################################
lr=LinearRegression()
lr.fit(ukpVal,maasVal)




#POLINOMIAL REGRESSION####################################

#2. Dereceden ####################################
pol_reg=PolynomialFeatures(degree=2) #2. dereceden polinom denklemi oluştu
egitimSeviyesiPoly=pol_reg.fit_transform(ukpVal)#b0+bx+bx2+bx3..... 1,2,9,16.... katsayılar oluştu
lr2=LinearRegression()
lr2.fit(egitimSeviyesiPoly,maasVal)



pol_reg=PolynomialFeatures(degree=3) #2. dereceden polinom denklemi oluştu
egitimSeviyesiPoly=pol_reg.fit_transform(ukpVal)#b0+bx+bx2+bx3..... 1,2,9,16.... katsayılar oluştu
lr2=LinearRegression()
lr2.fit(egitimSeviyesiPoly,maasVal)

#SUPPORT VECTOR REGRESSION##############################################################################################################
sc=StandardScaler()
egitimSeviyesiOlcekli=sc.fit_transform(ukpVal)

sc2=StandardScaler()
maasOlcekli=sc2.fit_transform(maasVal)

#Değerleri standart ölçekten geçirdik

from sklearn.svm import SVR
# KERNELS RBF,LİNEAR,POLİNOMİAL
svr_reg=SVR(kernel='rbf')
svr_reg.fit(egitimSeviyesiOlcekli,maasOlcekli)

svr_lin = SVR(kernel='linear')
svr_lin.fit(egitimSeviyesiOlcekli,maasOlcekli)



svr_poly = SVR(kernel='poly')
svr_poly.fit(egitimSeviyesiOlcekli,maasOlcekli)


#DECISION TREE REGRESSION##############################################################################################################
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(ukp,maas)


#RANDOM FOREST REGRESSION##############################################################################################################
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0,) #n_estimators kaç tane alt ağaca bölüneceğini belirler. random_state = 0 ifadesi, bize sonuçların tekrarlanması olanağını sağla
rf_reg.fit(ukp,maas)

#Decision Tree veri kümesindeki sonuçlardan birini döndürürken random forest daha esnek rakamlar tahmin edebilir.
#Bunu birden fazla decison tree nin verdiği kararların ortalamasını alarak yapar.



#R2 (R-SUARE) YÖNTEMİ#####################################################################################################################

#print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(yeniEs)))
#print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(yeniEs2)))
print('###################################################################################')
print('R2 SKORLARI: ')
print('LINEAR REGRESSİON R2 skoru: ',r2_score(maasVal,lr.predict(ukpVal)))
print('POLINOMIAL LINEAR REGRESSİON R2 skoru: ',r2_score(maasVal,lr2.predict(pol_reg.fit_transform(ukpVal))))
print('SUPPORT VECTOR REGRESSİON R2 skoru: ',r2_score(maasOlcekli,svr_poly.predict(egitimSeviyesiOlcekli)))# burada radyal değeri değiştirip tekrar bakmak sağlıklı olabilir
print('DECISION TREE REGRESSION R2 skoru: ',r2_score(maas,r_dt.predict(ukp)))#R square yüksek çıkar kanmayın
print('RANDOM FOREST REGRESSION R2 skoru: ',r2_score(maas,rf_reg.predict(ukp)))

print('')
print('')


print('###################################################################################')
print('3 PARAMETELI TAHMIN SONUCU')
print('UNVAN SEVİYESİ 7 KIDEMI 10 PUANI 100 OLACAK ŞEKİLDE TAHMİNLER: ')
print('LINEAR REGRESSİON: ',lr.predict(np.array([[7,10,100]])))
print('POLINOMIAL LINEAR REGRESSİON 3. DERECEDEN: ',lr2.predict(pol_reg.fit_transform(np.array([[7,10,100]]))))
print("Support Vektor Regression Linear : ",svr_lin.predict(np.array([[7,10,100]])))
print("Support Vektor Regression RBF : ",svr_reg.predict(np.array([[7,10,100]])))
print("Support Vektor Regression Poly : ",svr_poly.predict(np.array([[7,10,100]])))
print("DECISION TREE REGRESSION : ",r_dt.predict(np.array([[7,10,100]])))
print('RANDOM FOREST REGRESSION',rf_reg.predict(np.array([[7,10,100]])))


'''
En sağlıklı R2 değeri 3 parametreli datasetinde gözüküyo o yüzden 3 parametreli  kullanıldı.
2 veya 3 parametre hesaplaması backward elemination ile bakılarak r2_score ile hesaplanmıştır.

2 parametreli

LINEAR REGRESSİON R2 skoru:  0.043718913359342415
POLINOMIAL LINEAR REGRESSİON R2 skoru:  0.1189844424066473
SUPPORT VECTOR REGRESSİON R2 skoru:  -0.09948007834452244
DECISION TREE REGRESSION R2 skoru:  0.6166584246355375
RANDOM FOREST REGRESSION R2 skoru:  0.4095911664038612

3 parametreli
LINEAR REGRESSİON R2 skoru:  0.5857207050854021
POLINOMIAL LINEAR REGRESSİON R2 skoru:  0.9853596133928922
SUPPORT VECTOR REGRESSİON R2 skoru:  0.7512875071065472
DECISION TREE REGRESSION R2 skoru:  1.0
RANDOM FOREST REGRESSION R2 skoru:  0.9475498704400864

1 parametreli
LINEAR REGRESSİON R2 skoru:  0.5285811733746243
POLINOMIAL LINEAR REGRESSİON R2 skoru:  0.7921542810762359
SUPPORT VECTOR REGRESSİON R2 skoru:  0.4876873681071481
DECISION TREE REGRESSION R2 skoru:  0.8343186200100907
RANDOM FOREST REGRESSION R2 skoru:  0.8284081476481634
'''