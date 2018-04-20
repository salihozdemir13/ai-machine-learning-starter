from numpy import exp, array, random, dot, mean, abs
import numpy as np

"""
            input   output
example1    0 0 1     0
example2    1 1 1     1
example3    1 0 1     1
example4    0 1 1     0

situation   1 0 0     ? - (answer is 1 shhhhh :))
"""
# Bilgisayara verdiğimiz 4 farklı örnek durumda çıktıları öğrenmesini ve öğrendiği bilgilerle yeni durumunda vereceğimiz inputu doğru cevaplamasını bekliyoruz.

girdi = array([[0,0,1], [1,1,1], [1,0,1]])

gercekSonuc = array([[0,1,1]]).T # T Transpose

sinapsAgirlik = array([[1.0,1.0,1.0]]).T

for tekrar in range(1000): # range sayısı artırılınca sinaps ağırlık oranı otimize olacak
    hucreDegeri = dot(girdi,sinapsAgirlik) # dot product matris çarpım işlemi
    #print("Hücre değeri")
    #print(hucreDegeri)
    tahmin = 1 / (1 + exp(-hucreDegeri))
    #print("Tahmin")
    #print(tahmin)
    #print("Hata oranı")
    #print((gercekSonuc - tahmin) * tahmin * (1 - tahmin))
    sinapsAgirlik += dot(girdi.T, ((gercekSonuc - tahmin) * tahmin * (1 - tahmin)))
    #print("Sinaps Ağırlığı")
    #print(sinapsAgirlik)
    print(str(np.mean(np.abs(gercekSonuc-tahmin)))) # hata oranının nasıl düştüğünü gözlemleyebiliriz

print("Cevap")
print(1 / (1 + exp(-(dot(array([1,0,0]), sinapsAgirlik)))))