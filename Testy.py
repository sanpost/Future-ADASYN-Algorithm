import numpy as np
from scipy.stats import ttest_ind
import csv

#wczytanie danych
real_Accuracy_Future_Adasyn = np.load('FutureAdasyn Real Accuracy.npy')
real_Precision_Future_Adasyn = np.load('FutureAdasyn Real Precision.npy')
real_F1_Future_Adasyn = np.load('FutureAdasyn Real F1-score.npy')
real_Recall_Future_Adasyn = np.load('FutureAdasyn Real Recall.npy')

real_Accuracy_Adasyn = np.load('Adasyn Real Accuracy.npy')
real_Precision_dasyn = np.load('Adasyn Real Precision.npy')
real_F1_Adasyn = np.load('Adasyn Real F1-score.npy')
real_Recall_Adasyn = np.load('Adasyn Real Recall.npy')

real_Accuracy_BSmote = np.load('BorderlineSMOTE Real Accuracy.npy')
real_Precision_BSmote = np.load('BorderlineSMOTE Real Precision.npy')
real_F1_BSmote = np.load('BorderlineSMOTE Real F1-score.npy')
real_Recall_BSmote = np.load('BorderlineSMOTE Real Recall.npy')

real_Accuracy_Smote = np.load('SMOTE Real Accuracy.npy')
real_Precision_Smote = np.load('SMOTE Real Precision.npy')
real_F1_Smote = np.load('SMOTE Real F1-score.npy')
real_Recall_Smote = np.load('SMOTE Real Recall.npy')


synthetic_Accuracy_Future_Adasyn = np.load('FutureAdasyn Synthetic Accuracy.npy')
synthetic_Precision_Future_Adasyn = np.load('FutureAdasyn Synthetic Precision.npy')
synthetic_F1_Future_Adasyn = np.load('FutureAdasyn Synthetic F1-score.npy')
synthetic_Recall_Future_Adasyn = np.load('FutureAdasyn Synthetic Recall.npy')

synthetic_Accuracy_Adasyn = np.load('Adasyn Synthetic Accuracy.npy')
synthetic_Precision_dasyn = np.load('Adasyn Synthetic Precision.npy')
synthetic_F1_Adasyn = np.load('Adasyn Synthetic F1-score.npy')
synthetic_Recall_Adasyn = np.load('Adasyn Synthetic Recall.npy')

synthetic_Accuracy_BSmote = np.load('BorderlineSMOTE Synthetic Accuracy.npy')
synthetic_Precision_BSmote = np.load('BorderlineSMOTE Synthetic Precision.npy')
synthetic_F1_BSmote = np.load('BorderlineSMOTE Synthetic F1-score.npy')
synthetic_Recall_BSmote = np.load('BorderlineSMOTE Synthetic Recall.npy')

synthetic_Accuracy_Smote = np.load('SMOTE Synthetic Accuracy.npy')
synthetic_Precision_Smote = np.load('SMOTE Synthetic Precision.npy')
synthetic_F1_Smote = np.load('SMOTE Synthetic F1-score.npy')
synthetic_Recall_Smote = np.load('SMOTE Synthetic Recall.npy')

Stdreal = [["Metody/Metryki", "Dokladnosc", "Precyzja", "F1-score", "Recall"],
                   ["Adasyn", f"{round(np.mean(real_Accuracy_Adasyn), 4)} +/- {round(np.std(real_Accuracy_Adasyn), 4)}", f"{round(np.mean(real_Precision_dasyn), 4)} +/-  {round(np.std(real_Precision_dasyn), 4)}",
                    f"{round(np.mean(real_F1_Adasyn), 4)} +/- {round(np.std(real_F1_Adasyn), 4)}", f"{round(np.mean(real_Recall_Adasyn), 4)} +/-  {round(np.std(real_Recall_Adasyn), 4)}"],

                   ["Future Adasyn", f"{round(np.mean(real_Accuracy_Future_Adasyn), 4)} +/-  {round(np.std(real_Accuracy_Future_Adasyn), 4)}", f"{round(np.mean(real_Precision_Future_Adasyn), 4)} +/-  {round(np.std(real_Precision_Future_Adasyn), 4)}",
                    f"{round(np.mean(real_F1_Future_Adasyn), 4)} +/-  {round(np.std(real_F1_Future_Adasyn), 4)}", f"{round(np.mean(real_Recall_Future_Adasyn), 4)} +/-  {round(np.std(real_Recall_Future_Adasyn), 4)}"],

                   ["BorderlineSMOTE", f"{round(np.mean(real_Accuracy_BSmote), 4)} +/-  {round(np.std(real_Accuracy_BSmote), 4)}", f"{round(np.mean(real_Precision_BSmote), 4)} +/-  {round(np.std(real_Precision_BSmote), 4)}",
                    f"{round(np.mean(real_F1_BSmote), 4)} +/-  {round(np.std(real_F1_BSmote), 4)}", f"{round(np.mean(real_Recall_BSmote), 4)}  +/- {round(np.std(real_Recall_BSmote), 4)}"],

                   ["SMOTE", f"{round(np.mean(real_Accuracy_Smote), 4)} +/- {round(np.std(real_Accuracy_Smote), 4)}", f"{round(np.mean(real_Precision_Smote), 4)} +/- {round(np.std(real_Precision_Smote), 4)}",
                    f"{round(np.mean(real_F1_Smote), 4)} +/- {round(np.std(real_F1_Smote), 4)}", f"{round(np.mean(real_Recall_Smote), 4)} +/- {round(np.std(real_Recall_Smote), 4)}"]]

Stdsynthetic = [["Metody/Metryki", "Dokladnosc", "Precyzja", "F1-score", "Recall"],
                   ["Adasyn", f"{round(np.mean(synthetic_Accuracy_Adasyn), 4)} +/- {round(np.std(synthetic_Accuracy_Adasyn), 4)}", f"{round(np.mean(synthetic_Precision_dasyn), 4)} +/- {round(np.std(synthetic_Precision_dasyn), 4)}",
                    f"{round(np.mean(synthetic_F1_Adasyn), 4)} +/- {round(np.std(synthetic_F1_Adasyn), 4)}", f"{round(np.mean(synthetic_Recall_Adasyn), 4)} +/- {round(np.std(synthetic_Recall_Adasyn), 4)}"],

                   ["Future Adasyn", f"{round(np.mean(synthetic_Accuracy_Future_Adasyn), 4)} +/- {round(np.std(synthetic_Accuracy_Future_Adasyn), 4)}", f"{round(np.mean(synthetic_Precision_Future_Adasyn), 4)} +/- {round(np.std(synthetic_Precision_Future_Adasyn), 4)}",
                    f"{round(np.mean(synthetic_F1_Future_Adasyn), 4)} +/- {round(np.std(synthetic_F1_Future_Adasyn), 4)}", f"{round(np.mean(synthetic_Recall_Future_Adasyn), 4)} +/- {round(np.std(synthetic_Recall_Future_Adasyn), 4)}"],

                   ["BorderlineSMOTE", f"{round(np.mean(synthetic_Accuracy_BSmote), 4)} +/- {round(np.std(synthetic_Accuracy_BSmote), 4)}", f"{round(np.mean(synthetic_Precision_BSmote), 4)} +/- {round(np.std(synthetic_Precision_BSmote), 4)}",
                    f"{round(np.mean(synthetic_F1_BSmote), 4)} +/- {round(np.std(synthetic_F1_BSmote), 4)}", f"{round(np.mean(synthetic_Recall_BSmote), 4)} +/- {round(np.std(synthetic_Recall_BSmote), 4)}"],

                   ["SMOTE", f"{round(np.mean(synthetic_Accuracy_Smote), 4)} +/- {round(np.std(synthetic_Accuracy_Smote), 4)}", f"{round(np.mean(synthetic_Precision_Smote), 4)} +/-  {round(np.std(synthetic_Precision_Smote), 4)}",
                    f"{round(np.mean(synthetic_F1_Smote), 4)} +/- {round(np.std(synthetic_F1_Smote), 4)}", f"{round(np.mean(synthetic_Recall_Smote), 4)} +/- {round(np.std(synthetic_Recall_Smote), 4)}"]]

with open('stdreal.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Description', 'Metric', 't-statistic', 'p-value'])
    writer.writerows(Stdreal)
  
with open('stdsynthetic.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Description', 'Metric', 't-statistic', 'p-value'])
    writer.writerows(Stdsynthetic)

# Testy t dla Future Adasyn vs inne metody dla danych rzeczywistych
real_future_adasyn_vs_adasyn = ttest_ind(real_Accuracy_Future_Adasyn, real_Accuracy_Adasyn)
real_future_adasyn_vs_bsmote = ttest_ind(real_Accuracy_Future_Adasyn, real_Accuracy_BSmote)
real_future_adasyn_vs_smote = ttest_ind(real_Accuracy_Future_Adasyn, real_Accuracy_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych syntetycznych
synthetic_future_adasyn_vs_adasyn = ttest_ind(synthetic_Accuracy_Future_Adasyn, synthetic_Accuracy_Adasyn)
synthetic_future_adasyn_vs_bsmote = ttest_ind(synthetic_Accuracy_Future_Adasyn, synthetic_Accuracy_BSmote)
synthetic_future_adasyn_vs_smote = ttest_ind(synthetic_Accuracy_Future_Adasyn, synthetic_Accuracy_Smote)

# Funkcja do zapisu wyników do pliku CSV
def save_results(filename, results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 't-statistic', 'p-value'])
        for row in results:
            writer.writerow(row)

# Testy t dla Adasyn vs inne metody dla danych rzeczywistych
real_results_accuracy = [
    ['Future Adasyn', round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_Future_Adasyn).statistic, 4), round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_BSmote).statistic, 4), round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_Smote).statistic, 4), round(ttest_ind(real_Accuracy_Adasyn, real_Accuracy_Smote).pvalue, 4)]
]
real_results_precision = [
    ['Future Adasyn', round(ttest_ind(real_Precision_dasyn, real_Precision_Future_Adasyn).statistic, 4), round(ttest_ind(real_Precision_dasyn, real_Precision_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(real_Precision_dasyn, real_Precision_BSmote).statistic, 4), round(ttest_ind(real_Precision_dasyn, real_Precision_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(real_Precision_dasyn, real_Precision_Smote).statistic, 4), round(ttest_ind(real_Precision_dasyn, real_Precision_Smote).pvalue, 4)]
]
real_results_f1 = [
    ['Future Adasyn', round(ttest_ind(real_F1_Adasyn, real_F1_Future_Adasyn).statistic, 4), round(ttest_ind(real_F1_Adasyn, real_F1_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(real_F1_Adasyn, real_F1_BSmote).statistic, 4), round(ttest_ind(real_F1_Adasyn, real_F1_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(real_F1_Adasyn, real_F1_Smote).statistic, 4), round(ttest_ind(real_F1_Adasyn, real_F1_Smote).pvalue, 4)]
]
real_results_recall = [
    ['Future Adasyn', round(ttest_ind(real_Recall_Adasyn, real_Recall_Future_Adasyn).statistic, 4), round(ttest_ind(real_Recall_Adasyn, real_Recall_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(real_Recall_Adasyn, real_Recall_BSmote).statistic, 4), round(ttest_ind(real_Recall_Adasyn, real_Recall_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(real_Recall_Adasyn, real_Recall_Smote).statistic, 4), round(ttest_ind(real_Recall_Adasyn, real_Recall_Smote).pvalue, 4)]
]

# Testy t dla Adasyn vs inne metody dla danych syntetycznych
synthetic_results_accuracy = [
    ['Future Adasyn', round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_Future_Adasyn).statistic, 4), round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_BSmote).statistic, 4), round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_Smote).statistic, 4), round(ttest_ind(synthetic_Accuracy_Adasyn, synthetic_Accuracy_Smote).pvalue, 4)]
]
synthetic_results_precision = [
    ['Future Adasyn', round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_Future_Adasyn).statistic, 4), round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_BSmote).statistic, 4), round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_Smote).statistic, 4), round(ttest_ind(synthetic_Precision_dasyn, synthetic_Precision_Smote).pvalue, 4)]
]
synthetic_results_f1 = [
    ['Future Adasyn', round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_Future_Adasyn).statistic, 4), round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_BSmote).statistic, 4), round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_Smote).statistic, 4), round(ttest_ind(synthetic_F1_Adasyn, synthetic_F1_Smote).pvalue, 4)]
]
synthetic_results_recall = [
    ['Future Adasyn', round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_Future_Adasyn).statistic, 4), round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_Future_Adasyn).pvalue, 4)],
    ['BorderlineSMOTE', round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_BSmote).statistic, 4), round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_BSmote).pvalue, 4)],
    ['SMOTE', round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_Smote).statistic, 4), round(ttest_ind(synthetic_Recall_Adasyn, synthetic_Recall_Smote).pvalue, 4)]
]

# Zapis wyników do plików CSV
save_results('real_adasyn_vs_other_accuracy.csv', real_results_accuracy)
save_results('real_adasyn_vs_other_precision.csv', real_results_precision)
save_results('real_adasyn_vs_other_f1.csv', real_results_f1)
save_results('real_adasyn_vs_other_recall.csv', real_results_recall)

save_results('synthetic_adasyn_vs_other_accuracy.csv', synthetic_results_accuracy)
save_results('synthetic_adasyn_vs_other_precision.csv', synthetic_results_precision)
save_results('synthetic_adasyn_vs_other_f1.csv', synthetic_results_f1)
save_results('synthetic_adasyn_vs_other_recall.csv', synthetic_results_recall)
