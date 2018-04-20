import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_csv("USD_TRY.csv")
# print(veri)

x = veri["Gun"]
y = veri["Fiyat"]

x = x.values.reshape(251, 1)
y = y.values.reshape(251, 1)

plt.scatter(x,y)

# Lineer Reg.
tahminLineer = LinearRegression()
tahminLineer.fit(x,y) # koordinatlara oturtma
tahminLineer.predict(x) # güne göre fiyat bulma, x eksenine göre y yi bul

plt.plot(x, tahminLineer.predict(x), c = "red")

# Polinom Reg.
tahminPolinom = PolynomialFeatures(degree=2)
xYeni = tahminPolinom.fit_transform(x) # yeni x matrisi oluştur, güne göre fit et.

polinomModel = LinearRegression()
polinomModel.fit(xYeni,y)
polinomModel.predict(xYeni)

plt.plot(x, polinomModel.predict(xYeni))
# plt.show()

hataKaresiLineer = 0
hataKaresiPolinom = 0

# Polinom Reg. Hatası
for i in range(len(xYeni)):
    hataKaresiPolinom = hataKaresiPolinom + (float(y[i]) - float(polinomModel.predict(xYeni)[i]))**2 # matris üzerinden işlem yapmasını engellemek için float kullandık. kullanmasaydık çıktı array([deger]) gibi bir çıktı üzerinden işlem yapacaktı.

# Lineer Reg. Hatası
for i in range(len(y)):
    hataKaresiLineer = hataKaresiLineer + (float(y[i]) - float(tahminLineer.predict(x)[i]))**2

"""
hataKaresiPolinom = 0

for a in range(150):

    tahminPolinom = PolynomialFeatures(degree=a+1)
    xYeni = tahminPolinom.fit_transform(x)

    polinomModel = LinearRegression()
    polinomModel.fit(xYeni,y)
    polinomModel.predict(xYeni)
    for i in range(len(xYeni)):
        hataKaresiPolinom = hataKaresiPolinom + (float(y[i])-float(polinomModel.predict(xYeni)[i]))**2
    print(a+1,"inci dereceden fonksiyonda hata,", hataKaresiPolinom)

    hataKaresiPolinom = 0
"""

# 8. dereceden polinom en doğru tahmini verir!
tahminPolinom8 = PolynomialFeatures(degree=8)
xYeni = tahminPolinom8.fit_transform(x)

polinomModel8 = LinearRegression()
polinomModel8.fit(xYeni,y)
polinomModel8.predict(xYeni)

plt.plot(x,polinomModel8.predict(xYeni))
plt.show()

print((float(y[201])-float(polinomModel8.predict(xYeni)[201]))) # 201. gün verisi