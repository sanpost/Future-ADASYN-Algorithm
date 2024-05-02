# Tworzenie danych syntetycznych i rzeczywistych, a następnie zastosowanie algorytmów oversamplingu FutureAdasyn, ADASYN, SMOTE oraz BorderlineSMOTE.
#  Wizualizacja danych przed i po oversamplingu.

from collections import Counter
import numpy as np
from FutureAdasyn import FutureAdasyn
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt

# Generuje dane syntetyczne
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_repeated=0, n_redundant=0, random_state=2137, weights=[0.1, 0.9])

fig, ax = plt.subplots(1, 1, figsize=(10, 10/1.414))
ax.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('original_data.png')

# Wyświetla ile próbek jest w klasie 0 (mniejszościowej) a ile w klasie 1 (większościowej)
print("\nLiczba próbek wygenerowanych przed oversamplingiem: ", Counter(y))

# Przykład użycia klasy FutureAdasyn

fads = FutureAdasyn()
ads = ADASYN()
smt = SMOTE()
bsmt = BorderlineSMOTE()

# Za pomocą zaimplementowanego algorytmu "fas" do klasy FutureAdasyn generuje nowe próbki

X_resampled_future, y_resampled_future = fads.fit_resample(X, y)
X_resampled, y_resampled = ads.fit_resample(X, y)
X_resampled_smt, y_resampled_smt = smt.fit_resample(X, y)
X_resampled_bsmt, y_resampled_bsmt = bsmt.fit_resample(X, y)

print("Liczba próbek syntetycznych po oversamplingu FutureAdasyn: ", Counter(y_resampled_future))
print("Liczba próbek syntetycznych po oversamplingu ADASYN: ", Counter(y_resampled))
print("Liczba próbek syntetycznych po oversamplingu SMOTE: ", Counter(y_resampled_smt))
print("Liczba próbek syntetycznych po oversamplingu BorderlineSMOTE: ", Counter(y_resampled_bsmt))

# Generuje obrazy dla czterech algorytmów oversamplingu

fig, axs = plt.subplots(2, 2, figsize=(15, 15/1.414))

methods = ['FutureAdasyn', 'ADASYN', 'SMOTE', 'BorderlineSMOTE']
resampled_data = [(X_resampled_future, y_resampled_future), (X_resampled, y_resampled), (X_resampled_smt, y_resampled_smt), (X_resampled_bsmt, y_resampled_bsmt)]

for i, method in enumerate(methods):
    axs[i//2, i%2].scatter(resampled_data[i][0][:, 0], resampled_data[i][0][:, 1], c=resampled_data[i][1])
    axs[i//2, i%2].set_title(method)
    axs[i//2, i%2].set_xlabel('Feature 1')
    axs[i//2, i%2].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('oversampled_data.png')

# Generuje dane rzeczywiste z pliku
real_data = np.loadtxt('real_data.csv', delimiter=';')
real_X = real_data[:, :-1]
real_y = real_data[:, -1]

# Generuje obraz dla danych rzeczywistych
plt.figure(figsize=(10, 10/1.414))
plt.scatter(real_X[:, 0], real_X[:, 1], c=real_y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.savefig('original_real_data.png')

# Wyświetla ile próbek jest w klasie 0 (mniejszościowej) a ile w klasie 1 (większościowej)
print("\nLiczba próbek z rzeczywistego zbioru danych: ", Counter(real_y))

# Przykład użycia klasy FutureAdasyn dla danych rzeczywistych

fads = FutureAdasyn()
ads = ADASYN()
smt = SMOTE()
bsmt = BorderlineSMOTE()

# Za pomocą zaimplementowanego algorytmu "fas" do klasy FutureAdasyn generuje nowe próbki dla danych rzeczywistych

real_X_resampled_future, real_y_resampled_future = fads.fit_resample(real_X, real_y)
real_X_resampled, real_y_resampled = ads.fit_resample(real_X, real_y)
real_X_resampled_smt, real_y_resampled_smt = smt.fit_resample(real_X, real_y)
real_X_resampled_bsmt, real_y_resampled_bsmt = bsmt.fit_resample(real_X, real_y)

print("Liczba próbek po oversamplingu FutureAdasyn dla danych rzeczywistych: ", Counter(real_y_resampled_future))
print("Liczba próbek po oversamplingu ADASYN dla danych rzeczywistych: ", Counter(real_y_resampled))
print("Liczba próbek po oversamplingu SMOTE dla danych rzeczywistych: ", Counter(real_y_resampled_smt))
print("Liczba próbek po oversamplingu BorderlineSMOTE dla danych rzeczywistych: ", Counter(real_y_resampled_bsmt))

# Generuje obrazy dla czterech algorytmów oversamplingu dla danych rzeczywistych

real_fig, real_axs = plt.subplots(2, 2, figsize=(15, 15/1.414))

methods = ['FutureAdasyn', 'ADASYN', 'SMOTE', 'BorderlineSMOTE']
resampled_data = [(real_X_resampled_future, real_y_resampled_future), (real_X_resampled, real_y_resampled), (real_X_resampled_smt, real_y_resampled_smt), (real_X_resampled_bsmt, real_y_resampled_bsmt)]

for i, method in enumerate(methods):
    real_axs[i//2, i%2].scatter(resampled_data[i][0][:, 0], resampled_data[i][0][:, 1], c=resampled_data[i][1])
    real_axs[i//2, i%2].set_title(f'{method} (Real Data)')
    real_axs[i//2, i%2].set_xlabel('Feature 1')
    real_axs[i//2, i%2].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('oversampled_real_data.png')

