from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import timeit
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier

import warnings
warnings.filterwarnings('ignore')

file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_path)


print(" ---------- STRUKTURA KOLUMN ---------- ")
print(df.dtypes)
print("-" * 30)

print("\n ----------  INFO ----------  ")
df.info()

print("\n ----------  STATYSTYKI OPISOWE ----------  ")
print(df.describe())

print("\n ----------  PIERWSZE WIERSZE ----------  ")
print(df.head())

print("----- MISSING DATA -----")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() else "Wszystko jest")
print()

#Duplitakty
print("----- DUPLICATES -----")
duplicates_count = df.duplicated().sum()
print(f"Liczba zduplikowanych wierszy: {duplicates_count}")
print()

print("----- OUTLIERS -----")
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    mid_range = Q3 - Q1
    lower_bound = Q1 - 1.5 * mid_range
    upper_bound = Q3 + 1.5 * mid_range

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: znaleziono {len(outliers)} outlierów (zakres: {lower_bound:.2f} - {upper_bound:.2f})")

df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
kolejnosc = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

df['CALC'] = df['CALC'].map(kolejnosc)
df['CAEC'] = df['CAEC'].map(kolejnosc)
df = pd.get_dummies(df, columns=['MTRANS'], dtype=int)

#ta ważna 
df['NObeyesdad'] = df['NObeyesdad'].map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6})

cols_to_corr = df.select_dtypes(include=['number']).columns
corr_matrix = df[cols_to_corr].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Macierz korelacji cech NEO')
plt.savefig("Macierz korelacji.png")


X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print(f"Rozmiar zbioru treningowego: {X_train_scaled.shape[0]} probek")
print(f"Rozmiar zbioru testowego: {X_test_scaled.shape[0]} probek")

dt_model = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)


total_time = timeit.timeit(lambda: dt_model.predict(X_test), number=10)
inference_time_avg = total_time / 10

propability = dt_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_dt = dt_model.predict(X_test)

print("Raport dla drzewa\n")
print(classification_report(y_test, y_pred_dt))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_dt,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(24, 12))
plot_tree(dt_model, feature_names=X.columns, class_names=True, filled=True, rounded=True, fontsize=10)
plt.title('Reguły decyzyjne', fontsize=20, pad=20)
plt.savefig("Reguły dec drzewa.png")

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek drzewo')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek.png")


cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metryki = ['accuracy', 'f1_weighted', 'recall_weighted']

cv_results = cross_validate(dt_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False)

print("--- Średnie wyniki z 5 foldów ---")
print(f"Średnie Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results['fit_time'].mean():.4f} sekundy")


print(f"Czas predykcji   : {inference_time_avg}")


log_reg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'newton-cg']
}

grid_search_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    scoring='recall_weighted',
    n_jobs=-1
)

print("Spr parametrow tutaj sie wiesza")
grid_search_lr.fit(X_train_scaled, y_train)

print("\n--- Wyniki Regresji Logistycznej ---")
print("Najlepsze parametry:", grid_search_lr.best_params_)
print(f"Najlepszy wynik z walidacji krzyżowej (Recall): {grid_search_lr.best_score_:.4f}")

best_lr_model = grid_search_lr.best_estimator_

total_time = timeit.timeit(lambda: best_lr_model.predict(X_test), number=10)
inference_time_avg_regresja = total_time / 10

propability = best_lr_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_lr = best_lr_model.predict(X_test_scaled)

print("Raport dla regresji\n")
print(classification_report(y_test, y_pred_lr))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_lr,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek regresja')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek regresja.png")


cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metryki = ['accuracy', 'f1_weighted', 'recall_weighted']

cv_results = cross_validate(best_lr_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False)

print("--- Średnie wyniki z 5 foldów ---")
print(f"Średnie Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results['fit_time'].mean():.4f} sekundy")

print(f"Czas predykcji regresji   : {inference_time_avg_regresja}")


# BAGGING


bagging_model = BaggingClassifier(
    estimator=dt_model,
    n_estimators=25,
    random_state=42,
    n_jobs=-1
)

bagging_model.fit(X_train, y_train)

total_time_bagging = timeit.timeit(lambda: bagging_model.predict(X_test), number=10)
inference_time_avg_bagging = total_time_bagging / 10

propability = bagging_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_bag = bagging_model.predict(X_test)

print("Raport dla modelu Bagging\n")
print(classification_report(y_test, y_pred_bag))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_bag,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_bag), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek Bagging')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek bagging.png")

cv_results_bagging = cross_validate(bagging_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (Bagging) ---")
print(f"Średnie Accuracy: {cv_results_bagging['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji Baggingu   : {inference_time_avg_bagging:.6f} sekundy")


bagging_model_regresja = BaggingClassifier(
    estimator=best_lr_model,
    n_estimators=15, #brak zmian daje 15 zeby czas byl sensowny
    random_state=42,
    n_jobs=-1
)

bagging_model_regresja.fit(X_train, y_train)

total_time_bagging = timeit.timeit(lambda: bagging_model_regresja.predict(X_test), number=10)
inference_time_avg_bagging = total_time_bagging / 10

propability = bagging_model_regresja.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_bag = bagging_model_regresja.predict(X_test)

print("Raport dla modelu Bagging regresja\n")
print(classification_report(y_test, y_pred_bag))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_bag,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_bag), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek Bagging')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek bagging regresja.png")

cv_results_bagging_regresja = cross_validate(bagging_model_regresja, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (Bagging regresja) ---")
print(f"Średnie Accuracy: {cv_results_bagging_regresja['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging_regresja['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging_regresja['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging_regresja['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji Baggingu   : {inference_time_avg_bagging:.6f} sekundy")

#ADA boosting

smart_stump = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
adaboost_model = AdaBoostClassifier(estimator=smart_stump, n_estimators=50, random_state=42)

adaboost_model.fit(X_train, y_train)

total_time_ada = timeit.timeit(lambda: adaboost_model.predict(X_test), number=10)
inference_time_avg_ada = total_time_ada / 10

propability = adaboost_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_ada = adaboost_model.predict(X_test)

print("Raport dla modelu AdaBoost plytkie drzewo\n")
print(classification_report(y_test, y_pred_ada))

print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_ada,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_ada), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek adaboost')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek adaboost.png")

cv_results_bagging_regresja = cross_validate(adaboost_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (adaboost) ---")
print(f"Średnie Accuracy: {cv_results_bagging_regresja['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging_regresja['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging_regresja['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging_regresja['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji adaboost   : {inference_time_avg_ada:.6f} sekundy")


base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

adaboost_lr_model = AdaBoostClassifier(
    estimator=base_lr,
    n_estimators=15,
    random_state=42
)

adaboost_lr_model.fit(X_train_scaled, y_train)

total_time_ada_lr = timeit.timeit(lambda: adaboost_lr_model.predict(X_test_scaled), number=10)
inference_time_avg_ada_lr = total_time_ada_lr / 10

propability = adaboost_lr_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_ada_lr = adaboost_lr_model.predict(X_test_scaled)

print("Raport dla modelu AdaBoost regresja\n")
print(classification_report(y_test, y_pred_ada_lr))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_ada_lr,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_ada_lr), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)',
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek adaboost')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek adaboost regresja.png")

cv_results_bagging_regresja = cross_validate(adaboost_lr_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (adaboost) ---")
print(f"Średnie Accuracy: {cv_results_bagging_regresja['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging_regresja['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging_regresja['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging_regresja['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji adaboost regresja   : {inference_time_avg_ada_lr:.6f} sekundy")


# GRADIENT god make it stop its 4 am i just wanna sleep its been 434 lines already


gb_model = GradientBoostingClassifier(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=3, 
    random_state=42
)

gb_model.fit(X_train, y_train)

total_time_gb = timeit.timeit(lambda: gb_model.predict(X_test), number=10)
inference_time_avg_gb = total_time_gb / 10

propability = gb_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100
y_pred_gb = gb_model.predict(X_test)

print("Raport dla modelu gradient\n")
print(classification_report(y_test, y_pred_gb))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_gb,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

plt.figure(figsize=(32, 24))
ax = sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, fmt='d', cmap='Oranges', cbar=False)
ax.set_xticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)',
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
ax.set_yticklabels([
    'Niedowaga (0)', 
    'Chudy/Norma (1)', 
    'Nadwaga I (2)', 
    'Nadwaga II (3)', 
    'Otylosc I (4)', 
    'Otylosc II (5)', 
    'Otylosc III (6)'
])
plt.title('Macierz pomylek gradient')
plt.xlabel('Klasa przewidziana przez model', fontsize=14, labelpad=10)
plt.ylabel('Rzeczywista klasa w danych', fontsize=14, labelpad=10)
plt.savefig("Macierz pomyłek gradient.png")

metryki = ['accuracy', 'f1_weighted', 'recall_weighted']
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_bagging_regresja = cross_validate(gb_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (gradient) ---")
print(f"Średnie Accuracy: {cv_results_bagging_regresja['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging_regresja['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging_regresja['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging_regresja['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji gradient   : {inference_time_avg_gb:.6f} sekundy")


# comitet czemu przez c 

committee_model = VotingClassifier(
    estimators=[
        ('Drzewo', dt_model),
        ('Regresja', best_lr_model),
        ('Gradient', gb_model)
    ],
    voting='soft'
)

committee_model.fit(X_train_scaled, y_train)

total_time_gb = timeit.timeit(lambda: committee_model.predict(X_test), number=10)
inference_time_avg_gb = total_time_gb / 10

propability = committee_model.predict_proba(X_test)
confidence = np.max(propability, axis=1) * 100

y_pred_committee = committee_model.predict(X_test_scaled)

print("Raport dla modelu komitetu\n")
print(classification_report(y_test, y_pred_committee))
print("####################################################################")
confidence_df = pd.DataFrame({
    'Rzeczywista_Klasa': np.array(y_test),
    'Przewidziana_Klasa': y_pred_committee,
    'Pewnosc_Procentowa': np.round(confidence, 2)
})
print(confidence_df)

cv_results_bagging_regresja = cross_validate(gb_model, X, y, cv=cv_strategy, scoring=metryki, return_train_score=False, n_jobs=-1)

print("--- Średnie wyniki z 5 foldów (komitet) ---")
print(f"Średnie Accuracy: {cv_results_bagging_regresja['test_accuracy'].mean():.4f}")
print(f"Średnie Recall: {cv_results_bagging_regresja['test_recall_weighted'].mean():.4f}")
print(f"Średni F1-score (weighted): {cv_results_bagging_regresja['test_f1_weighted'].mean():.4f}")
print(f"Średni czas trenowania jednego foldu: {cv_results_bagging_regresja['fit_time'].mean():.4f} sekundy")

print(f"\nCzas predykcji komitet   : {inference_time_avg_gb:.6f} sekundy")