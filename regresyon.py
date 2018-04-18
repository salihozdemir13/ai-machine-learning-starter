# matematik kütüphanesi, numerical python
import numpy as np
# veri çekmek, gerekli veriyi okumak için kullanılır
import pandas as pd
from sklearn.linear_model import LinearRegression
# veriyi çizdirmek için kullanılır
import matplotlib.pyplot as plt

# veriyi okuyoruz
data = pd.read_csv("test.csv")

print(data)

x = data["x"]
y = data["y"]
x = pd.DataFrame.as_matrix(x) # Numpy matrislerine dönüştür
y = pd.DataFrame.as_matrix(y) # Numpy matrislerine dönüştür

print(x)
print(y)

plt.scatter(x,y) # 2 boyutlu grafikte oluşturduğumuza bakalım

m,b = np.polyfit(x, y, 1) # Numpy çizgimizi grafiğe oturtur (x eksen, y eksen, 1 kaçıncı dereceden polinom denklemi)

a = np.arange(150) # denklem hazırlandıktan sonra a nın aralığı ayarlanır

plt.scatter(x,y) # 2 nokta çizdirimi
plt.plot(m*a+b)

z = int(input("Kaç metre kare giriniz? : "))
tahmin = m * z + b
print(tahmin)

plt.scatter(z, tahmin, c = "red", marker = ">")
plt.show()
print("y = ", m, "x + ", b)



