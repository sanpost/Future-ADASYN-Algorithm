from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter

class FutureAdasyn(BaseOverSampler):

    def __init__(self): # Definiuje konstruktor klasy.
        super().__init__()
    
    def _fit_resample(self, x, y, K=5, threshold=1):

        # Zlicza liczbę wystąpień poszczególnych klas.
        numberOfSamples = Counter(y)
        # print("Number of samples: ", numberOfSamples)
        # Znajduje liczbę próbek w klasie mniejszościowej.
        minorityClassSamples = min(numberOfSamples.values())
        # print("Minority class samples: ", minorityClassSamples)
        # Znajduje liczbę próbek w klasie większościowej.
        majorityClassSamples = max(numberOfSamples.values())
        # print("Majority class samples: ", majorityClassSamples)
        # Oblicza liczbę próbek do dodania do klasy mniejszościowej.
        samplesToAdd = int((threshold * majorityClassSamples - minorityClassSamples))
        # print("Samples to add: ", samplesToAdd)
        # Tworzy listę próbek klasy mniejszościowej.
        minorityClass = np.where(y == 0)[0]
        listOfMinorityClassSamples = x[minorityClass]
        # print("List of minority class samples: ", listOfMinorityClassSamples)
        # Używa algorytmu najbliższych sąsiadów.
        nbrs = NearestNeighbors(n_neighbors=K).fit(listOfMinorityClassSamples)
        nbrsOfSample = nbrs.kneighbors(listOfMinorityClassSamples, return_distance=False)

        # Generuje próbki w klasie mniejszościowej.
        for i in range(samplesToAdd):
            randomSample = np.random.randint(len(minorityClass))
            randomSampleNBRS = np.random.choice(nbrsOfSample[randomSample])
            syntheticSample = listOfMinorityClassSamples[randomSample] + np.random.rand() * (
                    listOfMinorityClassSamples[randomSampleNBRS] - listOfMinorityClassSamples[randomSample])

            x = np.concatenate((x, np.array([syntheticSample])))
            y = np.concatenate((y, np.full(1, 0)))

        # Zwraca zbiór danych po zastosowaniu algorytmu.
        return x, y
    