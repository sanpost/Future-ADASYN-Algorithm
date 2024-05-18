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

with open('real_future_adasyn_vs_other.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', real_future_adasyn_vs_adasyn.statistic, real_future_adasyn_vs_adasyn.pvalue])
    writer.writerow(['BorderlineSMOTE', real_future_adasyn_vs_bsmote.statistic, real_future_adasyn_vs_bsmote.pvalue])
    writer.writerow(['SMOTE', real_future_adasyn_vs_smote.statistic, real_future_adasyn_vs_smote.pvalue])

with open('synthetic_future_adasyn_vs_other.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', synthetic_future_adasyn_vs_adasyn.statistic, synthetic_future_adasyn_vs_adasyn.pvalue])
    writer.writerow(['BorderlineSMOTE', synthetic_future_adasyn_vs_bsmote.statistic, synthetic_future_adasyn_vs_bsmote.pvalue])
    writer.writerow(['SMOTE', synthetic_future_adasyn_vs_smote.statistic, synthetic_future_adasyn_vs_smote.pvalue])

    # Testy t dla Future Adasyn vs inne metody dla danych rzeczywistych - Precyzja
real_future_adasyn_vs_adasyn_prec = ttest_ind(real_Precision_Future_Adasyn, real_Precision_dasyn)
real_future_adasyn_vs_bsmote_prec = ttest_ind(real_Precision_Future_Adasyn, real_Precision_BSmote)
real_future_adasyn_vs_smote_prec = ttest_ind(real_Precision_Future_Adasyn, real_Precision_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych rzeczywistych - F1
real_future_adasyn_vs_adasyn_f1 = ttest_ind(real_F1_Future_Adasyn, real_F1_Adasyn)
real_future_adasyn_vs_bsmote_f1 = ttest_ind(real_F1_Future_Adasyn, real_F1_BSmote)
real_future_adasyn_vs_smote_f1 = ttest_ind(real_F1_Future_Adasyn, real_F1_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych rzeczywistych - Recall
real_future_adasyn_vs_adasyn_recall = ttest_ind(real_Recall_Future_Adasyn, real_Recall_Adasyn)
real_future_adasyn_vs_bsmote_recall = ttest_ind(real_Recall_Future_Adasyn, real_Recall_BSmote)
real_future_adasyn_vs_smote_recall = ttest_ind(real_Recall_Future_Adasyn, real_Recall_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych syntetycznych - Precyzja
synthetic_future_adasyn_vs_adasyn_prec = ttest_ind(synthetic_Precision_Future_Adasyn, synthetic_Precision_dasyn)
synthetic_future_adasyn_vs_bsmote_prec = ttest_ind(synthetic_Precision_Future_Adasyn, synthetic_Precision_BSmote)
synthetic_future_adasyn_vs_smote_prec = ttest_ind(synthetic_Precision_Future_Adasyn, synthetic_Precision_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych syntetycznych - F1
synthetic_future_adasyn_vs_adasyn_f1 = ttest_ind(synthetic_F1_Future_Adasyn, synthetic_F1_Adasyn)
synthetic_future_adasyn_vs_bsmote_f1 = ttest_ind(synthetic_F1_Future_Adasyn, synthetic_F1_BSmote)
synthetic_future_adasyn_vs_smote_f1 = ttest_ind(synthetic_F1_Future_Adasyn, synthetic_F1_Smote)

# Testy t dla Future Adasyn vs inne metody dla danych syntetycznych - Recall
synthetic_future_adasyn_vs_adasyn_recall = ttest_ind(synthetic_Recall_Future_Adasyn, synthetic_Recall_Adasyn)
synthetic_future_adasyn_vs_bsmote_recall = ttest_ind(synthetic_Recall_Future_Adasyn, synthetic_Recall_BSmote)
synthetic_future_adasyn_vs_smote_recall = ttest_ind(synthetic_Recall_Future_Adasyn, synthetic_Recall_Smote)

# Zapis wyników do plików CSV dla danych rzeczywistych - Precyzja
with open('real_future_adasyn_vs_other_precision.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(real_future_adasyn_vs_adasyn_prec.statistic, 4), round(real_future_adasyn_vs_adasyn_prec.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(real_future_adasyn_vs_bsmote_prec.statistic, 4), round(real_future_adasyn_vs_bsmote_prec.pvalue, 4)])
    writer.writerow(['SMOTE', round(real_future_adasyn_vs_smote_prec.statistic, 4), round(real_future_adasyn_vs_smote_prec.pvalue, 4)])

# Zapis wyników do plików CSV dla danych rzeczywistych - F1
with open('real_future_adasyn_vs_other_f1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(real_future_adasyn_vs_adasyn_f1.statistic, 4), round(real_future_adasyn_vs_adasyn_f1.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(real_future_adasyn_vs_bsmote_f1.statistic, 4), round(real_future_adasyn_vs_bsmote_f1.pvalue, 4)])
    writer.writerow(['SMOTE', round(real_future_adasyn_vs_smote_f1.statistic, 4), round(real_future_adasyn_vs_smote_f1.pvalue, 4)])

# Zapis wyników do plików CSV dla danych rzeczywistych - Recall
with open('real_future_adasyn_vs_other_recall.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(real_future_adasyn_vs_adasyn_recall.statistic, 4), round(real_future_adasyn_vs_adasyn_recall.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(real_future_adasyn_vs_bsmote_recall.statistic, 4), round(real_future_adasyn_vs_bsmote_recall.pvalue, 4)])
    writer.writerow(['SMOTE', round(real_future_adasyn_vs_smote_recall.statistic, 4), round(real_future_adasyn_vs_smote_recall.pvalue, 4)])

# Zapis wyników do plików CSV dla danych syntetycznych - Precyzja
with open('synthetic_future_adasyn_vs_other_precision.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(synthetic_future_adasyn_vs_adasyn_prec.statistic, 4), round(synthetic_future_adasyn_vs_adasyn_prec.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(synthetic_future_adasyn_vs_bsmote_prec.statistic, 4), round(synthetic_future_adasyn_vs_bsmote_prec.pvalue, 4)])
    writer.writerow(['SMOTE', round(synthetic_future_adasyn_vs_smote_prec.statistic, 4), round(synthetic_future_adasyn_vs_smote_prec.pvalue, 4)])

# Zapis wyników do plików CSV dla danych syntetycznych - F1
with open('synthetic_future_adasyn_vs_other_f1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(synthetic_future_adasyn_vs_adasyn_f1.statistic, 4), round(synthetic_future_adasyn_vs_adasyn_f1.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(synthetic_future_adasyn_vs_bsmote_f1.statistic, 4), round(synthetic_future_adasyn_vs_bsmote_f1.pvalue, 4)])
    writer.writerow(['SMOTE', round(synthetic_future_adasyn_vs_smote_f1.statistic, 4), round(synthetic_future_adasyn_vs_smote_f1.pvalue, 4)])

# Zapis wyników do plików CSV dla danych syntetycznych - Recall
with open('synthetic_future_adasyn_vs_other_recall.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 't-statistic', 'p-value'])
    writer.writerow(['Adasyn', round(synthetic_future_adasyn_vs_adasyn_recall.statistic, 4), round(synthetic_future_adasyn_vs_adasyn_recall.pvalue, 4)])
    writer.writerow(['BorderlineSMOTE', round(synthetic_future_adasyn_vs_bsmote_recall.statistic, 4), round(synthetic_future_adasyn_vs_bsmote_recall.pvalue, 4)])
    writer.writerow(['SMOTE', round(synthetic_future_adasyn_vs_smote_recall.statistic, 4), round(synthetic_future_adasyn_vs_smote_recall.pvalue, 4)])
