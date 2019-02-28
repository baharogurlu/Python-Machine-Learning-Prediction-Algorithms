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






















