#Kollektif öğrenme bütün algoritmalar kullanılarak hata seviyesi minimuma indirilerek işlem yapılmaya çalışılır.
#Random forest train setini birden fazla kümeye bölerek birden fazla tahmşn çalışması yürütülür ve mojority voting (çoğunluğun oyu) yöntemi ile tahminde bulunr.
#Tahmin sonuçlarının ortalaması alınıyo denilebilir.
#Neden birden fazla kümeye bölünüyor?
#Karar ağlaçları veri arttıkça ezberlemeye başlar veya ağaç boyutu büyüdükçe hesaplama yüküde artıp yavaşlama olabilir.

from sklearn.ensemble import  RandomForestRegressor
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