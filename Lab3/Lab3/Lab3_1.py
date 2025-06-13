# -*- coding: cp1251 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
data = pd.read_csv('C:/Users/avazb/Desktop/BigData/Lab3/train.csv')

# Балансировка классов (ровное количество 0 и 1)
class_0 = data[data['Class'] == 0]
class_1 = data[data['Class'] == 1]
min_samples = min(len(class_0), len(class_1))

balanced_data = pd.concat([
    class_0.sample(min_samples, random_state=42),
    class_1.sample(min_samples, random_state=42)
])

print("Количество данных до сокращения: 0:  ", class_0.size, ".   1: ", class_1.size)
print("Количество данных (0 и 1) после сокращения:", balanced_data.size)

# Разделение на признаки и целевую переменную
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение модели (используем RandomForest как один алгоритм классификации)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Предсказание и оценка качества
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)