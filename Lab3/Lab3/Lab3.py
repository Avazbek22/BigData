# -*- coding: cp1251 -*-

import pandas as pd

# Загрузка данных
data = pd.read_csv('C:/Users/avazb/Desktop/BigData/Lab3/train.csv')

# A. Сколько строк в датафрейме, сколько столбцов
num_rows, num_cols = data.shape
print(f"Количество строк: {num_rows}, Количество столбцов: {num_cols}")

# B. Сколько места занимает датафрейм в оперативной памяти
memory_usage = data.memory_usage(deep=True).sum()
print(f"Память, занимаемая датафреймом: {memory_usage} байт")

# C. Для каждой интервальной переменной подсчитать мин, медиану, среднее, макс и персентили 25, 75
interval_columns = data.select_dtypes(include=['float64', 'int64']).columns
interval_stats = data[interval_columns].describe(percentiles=[.25, .5, .75])
print(interval_stats)

# D. Для каждой категориальной переменной рассчитать моду и сколько раз мода встречается в данных
if 'Class' in data.columns:
    # Преобразуем в категориальный тип, если это числовой столбец с дискретными значениями
    if data['Class'].dtype in ['int64', 'float64']:
        data['Class'] = data['Class'].astype('category')
    
    mode = data['Class'].mode()[0]
    mode_count = data['Class'].value_counts().iloc[0]
    print(f"Мода для Class: {mode}, Количество: {mode_count}")
    
    # Дополнительная статистика по категориям
    class_counts = data['Class'].value_counts()
    
else:
    print("Столбец 'Class' не найден в данных")


print("\nРаспределение классов:")
print(class_counts)


# A. Анализ и обработка пропусков
print("\nПропуски:")
print(data.isnull().sum())  # Проверка на пропуски

# C. Анализ и обработка категориальных переменных
categorical_columns = data.select_dtypes(include=['category']).columns
print(f"Количество категориальных переменных: {len(categorical_columns)}")

# Применение one-hot encoding
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# D. Разделить датасет на трейн и тест
from sklearn.model_selection import train_test_split

X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# A. KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"KNN Accuracy: {knn_accuracy}")

# B. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")

# C. SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy}")

# Сравнение точности
accuracies = {
    'KNN': knn_accuracy,
    'Logistic Regression': log_reg_accuracy,
    'SVM': svm_accuracy
}

best_model = max(accuracies, key=accuracies.get)
print(f"Лучший алгоритм: {best_model} с точностью {accuracies[best_model]}")