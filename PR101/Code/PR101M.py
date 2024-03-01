"""
Machine Learning Project: PR 101

Authors: Mattia Palmiotto; Riccardo Petrella
"""

from genetic_selection import GeneticSelectionCV                  # 0.5.1
import matplotlib.pyplot as plt                                   # 3.5.1 3.6.2
import numpy as np                                                # 1.21.5
import pandas as pd                                               # 1.4.2 1.4.4
from scipy.io.arff import loadarff                                # 1.7.3 1.9.3
import seaborn as sns                                             # 0.11.2
from sklearn.linear_model import LogisticRegression               # 1.0.2
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

DATA_PATH = r'...\PR101\Code\data'

#######################################

raw_data = loadarff(f'{DATA_PATH}\CM1.arff')
cm1_df = pd.DataFrame(raw_data[0])
cm1_df.shape

cm1_df.isnull().sum().sum()
cm1_df.info

#######################################

categorical = [i for i in cm1_df.columns if cm1_df[i].dtype == 'O']

print('There are {} categorical variables.'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical, '\n')

numerical = [i for i in cm1_df.columns if cm1_df[i].dtype in ['float64', 'int64']]

print('There are {} numerical variables.'.format(len(numerical)))

print('The numerical variables are: \n\n', numerical)

x = cm1_df.drop(['Defective'], axis=1)
y = cm1_df['Defective']

y = y.replace(b'N', 0)
y = y.replace(b'Y', 1)

#######################################

scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=numerical)

#######################################

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0)

#######################################

#### NO FS

# NB

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

tr_NB = gnb.score(X_train, Y_train)

accuracy_NB = gnb.score(X_test, Y_test)

Y_pred_NB = gnb.predict(X_test)

# DT

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

tr_DT = dt.score(X_train, Y_train)

accuracy_DT = dt.score(X_test, Y_test)

Y_pred_DC = dt.predict(X_test)

# LR

lr = LogisticRegression()
lr.fit(X_train, Y_train)

tr_LR = lr.score(X_train, Y_train)

accuracy_LR = lr.score(X_test, Y_test)

Y_pred_LR = lr.predict(X_test)

# KNN

knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
knn.fit(X_train, Y_train)

tr_KNN = knn.score(X_train, Y_train)

accuracy_KNN = knn.score(X_test, Y_test)

Y_pred_KNN = knn.predict(X_test)

#

tr_NoFS = [tr_NB, tr_DT, tr_LR, tr_KNN]
accuracies_NoFS = [accuracy_NB, accuracy_DT, accuracy_LR, accuracy_KNN]

# MC1: Methods of Feature selection (FS)

# FS made with Filter Feature Ranking (FFR) - Information Gain (IG)

IG = SelectKBest(mutual_info_classif, k=6)
X_IG_train = IG.fit_transform(X_train, Y_train)

IG_columns = []
numerical_lists = {col: list(X_train[col]) for col in numerical}

for i in range(6):
    list_i = list(X_IG_train[:, i])
    for col in numerical:
        if list_i == numerical_lists[col]:
            IG_columns.append(col)
            break

X_IG_train = pd.DataFrame(X_IG_train, columns=IG_columns)

X_IG_test = X_test[IG_columns]


def variation(first_value, second_value):
    return ((second_value - first_value) / first_value) * 100

# FFR:  IG + NB

gnb.fit(X_IG_train, Y_train)

tr_IG_NB = gnb.score(X_IG_train, Y_train)

accuracy_IG_NB = gnb.score(X_IG_test, Y_test)

Y_IG_pred_NB = gnb.predict(X_IG_test)

variation_IG_NB = variation(accuracy_NB, accuracy_IG_NB)

# FFR:  IG + DT

train_scores_IG_DT = []
accuracies_IG_DT = []

for i in range(50):

    dt.fit(X_IG_train, Y_train)

    train_scores_IG_DT.append(dt.score(X_IG_train, Y_train))

    accuracies_IG_DT.append(dt.score(X_IG_test, Y_test))

    Y_IG_pred_DC = dt.predict(X_IG_test)

tr_IG_DT = np.mean(train_scores_IG_DT)
accuracy_IG_DT = np.mean(accuracies_IG_DT)
variation_IG_DT = variation(accuracy_DT, accuracy_IG_DT)

# FFR: IG + LR

lr.fit(X_IG_train, Y_train)

tr_IG_LR = lr.score(X_IG_train, Y_train)

accuracy_IG_LR = lr.score(X_IG_test, Y_test)

Y_IG_pred_LR = lr.predict(X_IG_test)

variation_IG_LR = variation(accuracy_LR, accuracy_IG_LR)

# FFR: IG + KNN

knn.fit(X_IG_train, Y_train)

tr_IG_KNN = knn.score(X_IG_train, Y_train)

accuracy_IG_KNN = knn.score(X_IG_test, Y_test)

Y_IG_pred_KNN = knn.predict(X_IG_test)

variation_IG_KNN = variation(accuracy_KNN, accuracy_IG_KNN)

#

tr_IG = [tr_IG_NB, tr_IG_DT, tr_IG_LR, tr_IG_KNN]
accuracies_IG = [accuracy_IG_NB, accuracy_IG_DT,accuracy_IG_LR, accuracy_IG_KNN]

#######################################

cm = confusion_matrix(Y_test, Y_IG_pred_LR)

print('Confusion matrix\n\n', cm)
print('\nTrue Negatives(TN) = ', cm[0, 0])
print('\nTrue Positives(TP) = ', cm[1, 1])
print('\nFalse Negatives(FP) = ', cm[0, 1])
print('\nFalse Positives(FN) = ', cm[1, 0])

cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()

#######################################

# FS made with Feature Subest Selection (FSS) - SELECTION BASED ON CORRELATION (CFS)

corr = x.corr()
corr.head()

plt.figure(figsize=(20, 15))
sns.heatmap(corr, annot=True)

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                break
    return col_corr

corr_features = correlation(x, 0.75)

print("Number of correlated features:", len(corr_features))
print("\nName of correlated features:", corr_features)

X_train_noncorr = X_train.drop(corr_features, axis=1)  # features selected
X_test_noncorr = X_test.drop(corr_features, axis=1)

print("\nNumber of remaining features:", X_train_noncorr.shape[1])

X_noncorr = pd.concat([X_train_noncorr, X_test_noncorr], axis=0)

#######################################

# FSS: CFS + GS + NB

GS_gnb = GeneticSelectionCV(estimator=gnb, n_population=200, n_generations=5,
                            crossover_proba=0.6)

GS_gnb.fit(X_train_noncorr, Y_train)

tr_GS_NB = GS_gnb.score(X_train_noncorr, Y_train)

accuracy_GS_NB = GS_gnb.score(X_test_noncorr, Y_test)
variation_GS_NB = variation(accuracy_NB, accuracy_GS_NB)

# FSS: CFS + GS + DT

GS_dt = GeneticSelectionCV(estimator=dt, n_population=200, n_generations=5,
                           crossover_proba=0.6)

accuracies_GS_DT = []
training_scores_GS_DT = []

for i in range(4):
    GS_dt.fit(X_train_noncorr, Y_train)
    training_scores_GS_DT.append(GS_dt.score(X_train_noncorr, Y_train))
    accuracies_GS_DT.append(GS_dt.score(X_test_noncorr, Y_test))

tr_GS_DT = np.mean(training_scores_GS_DT)
accuracy_GS_DT = np.mean(accuracies_GS_DT)
variation_GS_DT = variation(accuracy_DT, accuracy_GS_DT)

# FSS: CFS + GS + LR

GS_lr = GeneticSelectionCV(estimator=lr, n_population=200, n_generations=5,
                           crossover_proba=0.6)

accuracies_GS_LR = []
training_scores_GS_LR = []

for i in range(4):
    GS_lr.fit(X_train_noncorr, Y_train)
    training_scores_GS_LR.append(GS_lr.score(X_train_noncorr, Y_train))
    accuracies_GS_LR.append(GS_lr.score(X_test_noncorr, Y_test))

tr_GS_LR = np.mean(training_scores_GS_LR)
accuracy_GS_LR = np.mean(accuracies_GS_LR)
variation_GS_LR = variation(accuracy_LR, accuracy_GS_LR)

# FSS: CFS + GS + KNN

GS_knn = GeneticSelectionCV(estimator=knn, n_population=200, n_generations=5,
                            crossover_proba=0.6)

accuracies_GS_KNN = []
training_scores_GS_KNN = []

for i in range(4):
    GS_knn.fit(X_train_noncorr, Y_train)
    training_scores_GS_KNN.append(GS_knn.score(X_train_noncorr, Y_train))
    accuracies_GS_KNN.append(GS_knn.score(X_test_noncorr, Y_test))

tr_GS_KNN = np.mean(training_scores_GS_KNN)
accuracy_GS_KNN = np.mean(accuracies_GS_KNN)
variation_GS_KNN = variation(accuracy_KNN, accuracy_GS_KNN)

#

tr_GS = [tr_GS_NB, tr_GS_DT, tr_GS_LR, tr_GS_KNN]
accuracies_GS = [accuracy_GS_NB, accuracy_GS_DT,accuracy_GS_LR, accuracy_GS_KNN]

### Results

acc_ind = ['No FS', 'IG (FRR)', 'GS (FSS)']

tr_table = {
    'NB': pd.Series([tr_NB, tr_IG_NB, tr_GS_NB], index=acc_ind),
    'DT': pd.Series([tr_DT, tr_IG_DT, tr_GS_DT], index=acc_ind),
    'LR': pd.Series([tr_LR, tr_IG_LR, tr_GS_LR], index=acc_ind),
    'KNN': pd.Series([tr_KNN, tr_IG_KNN, tr_GS_KNN], index=acc_ind)
}

tr_table = pd.DataFrame(tr_table)
tr_table = tr_table*100
print(tr_table)

accuracies_CM1 = {
    'NB': pd.Series([accuracy_NB, accuracy_IG_NB, accuracy_GS_NB], index=acc_ind),
    'DT': pd.Series([accuracy_DT, accuracy_IG_DT, accuracy_GS_DT], index=acc_ind),
    'LR': pd.Series([accuracy_LR, accuracy_IG_LR, accuracy_GS_LR], index=acc_ind),
    'KNN': pd.Series([accuracy_KNN, accuracy_IG_KNN, accuracy_GS_KNN], index=acc_ind)
}

accuracies_CM1 = pd.DataFrame(accuracies_CM1)
accuracies_CM1 = accuracies_CM1*100
print(accuracies_CM1)

variations_CM1 = {
    'NB': pd.Series([0, variation_IG_NB, variation_GS_NB], index=acc_ind),
    'DT': pd.Series([0, variation_IG_DT, variation_GS_DT], index=acc_ind),
    'LR': pd.Series([0, variation_IG_LR, variation_GS_LR], index=acc_ind),
    'KNN': pd.Series([0, variation_IG_KNN, variation_GS_KNN], index=acc_ind)
}

variations_CM1 = pd.DataFrame(variations_CM1)
print(variations_CM1)

#######################################

RESULTS_PATH = r'...\PR101\Code\results'
accuracies_KC1 = pd.read_csv(f"{RESULTS_PATH}/accuracies_KC1.csv", index_col=0)
accuracies_KC3 = pd.read_csv(f"{RESULTS_PATH}/accuracies_KC3.csv", index_col=0)
accuracies_MW1 = pd.read_csv(f"{RESULTS_PATH}/accuracies_MW1.csv", index_col=0)
accuracies_PC2 = pd.read_csv(f"{RESULTS_PATH}/accuracies_PC2.csv", index_col=0)
variations_KC1 = pd.read_csv(f"{RESULTS_PATH}/variations_KC1.csv", index_col=0)
variations_KC3 = pd.read_csv(f"{RESULTS_PATH}/variations_KC3.csv", index_col=0)
variations_MW1 = pd.read_csv(f"{RESULTS_PATH}/variations_MW1.csv", index_col=0)
variations_PC2 = pd.read_csv(f"{RESULTS_PATH}/variations_PC2.csv", index_col=0)

### Comparisons

accuracies_list = [accuracies_CM1, accuracies_KC1,accuracies_KC3, accuracies_MW1, accuracies_PC2]
variations_list = [variations_CM1, variations_KC1,variations_KC3, variations_MW1, variations_PC2]

datasets = ['CM1', 'KC1', 'KC3', 'MW1', 'PC2']
classifiers = ['NB', 'DT', 'LR', 'KNN']

for i in range(5):

    accuracies_table = accuracies_list[i]
    accuracies_NoFS = accuracies_table.loc['No FS']
    accuracies_IG = accuracies_table.loc['IG (FRR)']
    accuracies_GS = accuracies_table.loc['GS (FSS)']

    variations_table = variations_list[i]
    variations_IG = variations_table.loc['IG (FRR)']
    variations_GS = variations_table.loc['GS (FSS)']

    acc_plot = plt.plot
    plt.figure(figsize=(15, 7))
    acc_plot(classifiers, accuracies_NoFS, label="No FS", linestyle="-", marker='o')
    acc_plot(classifiers, accuracies_IG, label="IG", linestyle="-", marker='o')
    acc_plot(classifiers, accuracies_GS, label="CFS+GS", linestyle="-", marker='o')
    plt.legend()

    plt.xlabel('Prediction Models')
    plt.ylabel('Accuracy values')
    plt.ylim(60, 100)
    plt.title('Comparison on the accuracies - ' + datasets[i])
    plt.show()

    vrt_plot = plt.plot
    plt.figure(figsize=(15, 7))
    vrt_plot(classifiers, [0, 0, 0, 0], label="No FS", linestyle="-", marker='o')
    vrt_plot(classifiers, variations_IG, label="IG", linestyle="-", marker='o')
    vrt_plot(classifiers, variations_GS, label="CFS+GS", linestyle="-", marker='o')
    plt.legend()

    plt.xlabel('Prediction Models')
    plt.ylabel('Degree of Variation')
    plt.ylim(-30, 30)
    plt.title('Comparison on the variations - ' + datasets[i])
    plt.show()
    
### Stability

average_variations = {c: {} for c in classifiers} 
for c in classifiers:

    var_ig = [variations_list[i][c][1] for i in range(5)]
    var_gs = [variations_list[i][c][2] for i in range(5)]

    av_var_ig = np.mean(var_ig)
    av_var_gs = np.mean(var_gs)

    average_variations[c] = {'IG': av_var_ig, 'GS': av_var_gs}
average_variations = pd.DataFrame(average_variations)
average_variations

standard_deviations = {c: {} for c in classifiers} 
average_accuracies = {c: {} for c in classifiers} 
for c in classifiers:

    acc_nofs = [accuracies_list[i][c][0] for i in range(5)]
    acc_ig = [accuracies_list[i][c][1] for i in range(5)]
    acc_gs = [accuracies_list[i][c][2] for i in range(5)]

    sd_acc_nofs = np.std(acc_nofs)
    sd_acc_ig = np.std(acc_ig)
    sd_acc_gs = np.std(acc_gs)

    av_acc_nofs = np.mean(acc_nofs)
    av_acc_ig = np.mean(acc_ig)
    av_acc_gs = np.mean(acc_gs)

    standard_deviations[c] = {'No FS': sd_acc_nofs,'IG': sd_acc_ig, 'GS': sd_acc_gs}
    average_accuracies[c] = {'No FS': av_acc_nofs,'IG': av_acc_ig, 'GS': av_acc_gs}

average_accuracies = pd.DataFrame(average_accuracies)
standard_deviations = pd.DataFrame(standard_deviations)

#######################################

## Accuracies

no_fs_col_acc = average_accuracies.iloc[0,:]
ig_col_acc = average_accuracies.iloc[1,:]
gs_col_acc = average_accuracies.iloc[2,:]

fig = plt.figure()
X = np.arange(4)

ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, no_fs_col_acc, color = 'b', width = 0.25)
ax.bar(X + 0.25, ig_col_acc, color = 'g', width = 0.25)
ax.bar(X + 0.50, gs_col_acc, color = 'r', width = 0.25)


ax.set_xticks(ticks = X+0.25, labels = classifiers)
ax.set_ylabel('Average Accuracies')

plt.ylim(60,90)
ax.set_title('Average Accuracies by classifier on all datasets')
ax.legend(average_accuracies.index)

plt.show()

## Coeffs of variation

coeffs_variation = {c:(standard_deviations[c]/average_accuracies[c])*100 for c in classifiers}
coeffs_variation = pd.DataFrame(coeffs_variation)

no_fs_col = coeffs_variation.iloc[0,:]
ig_col = coeffs_variation.iloc[1,:]
gs_col = coeffs_variation.iloc[2,:]

fig = plt.figure()
X = np.arange(4)

ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, no_fs_col, color = 'b', width = 0.25)
ax.bar(X + 0.25, ig_col, color = 'g', width = 0.25)
ax.bar(X + 0.50, gs_col, color = 'r', width = 0.25)

ax.set_xticks(ticks = X+0.25, labels = classifiers)
ax.set_ylabel('Coeff. of Variations')
ax.set_title('Coefficients of Variations by classifier')
ax.legend(coeffs_variation.index)

plt.show()

#######################################

ave_acc_paper = pd.read_csv(f'{RESULTS_PATH}/ave_acc_paper.csv', index_col=0)

ave_var_paper = pd.read_csv(f'{RESULTS_PATH}/ave_var_paper.csv', index_col=0)

ave_training_scores = pd.read_csv(f'{RESULTS_PATH}/ave_tr_scores.csv', index_col=0)

print('\nAverage accuracies:\n')
print(average_accuracies)
print('\nAverage accuracies in the paper:\n')
print(ave_acc_paper)
print('\nAverage variations:\n')
print(average_variations)
print('\nAverage variations in the paper:\n')
print(ave_var_paper)
print('\nAverage training scores:\n')
print(ave_training_scores)
