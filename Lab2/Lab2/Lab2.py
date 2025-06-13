# -*- coding: cp1251 -*-

import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Подключение к базе данных SQLite
database = r'C:\Users\avazb\Desktop\BigData\Lab1\Formula 1 Race Data\Formula1.sqlite'
conn = sqlite3.connect(database)

# Загрузка данных из таблиц
races = pd.read_sql_query("SELECT * FROM races", conn)
drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
constructors = pd.read_sql_query("SELECT * FROM constructors", conn)
results = pd.read_sql_query("SELECT * FROM results", conn)
circuits = pd.read_sql_query("SELECT * FROM circuits", conn)

# Закрытие соединения с базой данных
conn.close()

# Приведение типа данных столбца 'constructorId' к int64 в таблице constructors
constructors['constructorId'] = constructors['constructorId'].astype('int64')

# Удаление дублирующихся столбцов перед объединением
if 'url' in circuits.columns:
    circuits = circuits.drop(columns=['url'])

# Объединение данных с указанием пользовательских суффиксов
df = pd.merge(results, races, on='raceId', how='left', suffixes=('_results', '_races'))
df = pd.merge(df, drivers, on='driverId', how='left', suffixes=('', '_drivers'))
df = pd.merge(df, constructors, on='constructorId', how='left', suffixes=('', '_constructors'))
df = pd.merge(df, circuits, on='circuitId', how='left', suffixes=('', '_circuits'))

# 1. Количество строк и столбцов
rows, cols = df.shape
print(f"Количество строк: {rows}, Количество столбцов: {cols}")

# 2. Разведочный анализ данных

# (a) Для числовых переменных
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Создаем пустой DataFrame для хранения результатов
numeric_summary = pd.DataFrame(index=numeric_columns, columns=[
    'Доля пропусков', 'Минимум', 'Максимум', 'Среднее', 'Медиана', 'Дисперсия', 
    'Квантиль 0.1', 'Квантиль 0.9', 'Квартиль 1', 'Квартиль 3'
])

# Заполняем DataFrame
for col in numeric_columns:
    numeric_summary.loc[col, 'Доля пропусков'] = df[col].isnull().mean()
    numeric_summary.loc[col, 'Минимум'] = df[col].min()
    numeric_summary.loc[col, 'Максимум'] = df[col].max()
    numeric_summary.loc[col, 'Среднее'] = df[col].mean()
    numeric_summary.loc[col, 'Медиана'] = df[col].median()
    numeric_summary.loc[col, 'Дисперсия'] = df[col].var()
    numeric_summary.loc[col, 'Квантиль 0.1'] = df[col].quantile(0.1)
    numeric_summary.loc[col, 'Квантиль 0.9'] = df[col].quantile(0.9)
    numeric_summary.loc[col, 'Квартиль 1'] = df[col].quantile(0.25)
    numeric_summary.loc[col, 'Квартиль 3'] = df[col].quantile(0.75)

print("Числовые переменные:")
print(numeric_summary)

# (b) Для категориальных переменных
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Создаем пустой DataFrame для хранения результатов
categorical_summary = pd.DataFrame(index=categorical_columns, columns=[
    'Доля пропусков', 'Количество уникальных значений', 'Мода'
])

# Заполняем DataFrame
for col in categorical_columns:
    categorical_summary.loc[col, 'Доля пропусков'] = df[col].isnull().mean()
    categorical_summary.loc[col, 'Количество уникальных значений'] = df[col].nunique()
    
    # Проверяем, есть ли данные в столбце
    if not df[col].empty and df[col].notna().any():
        categorical_summary.loc[col, 'Мода'] = df[col].mode()[0]  # Берем первую моду
    else:
        categorical_summary.loc[col, 'Мода'] = np.nan  # Если данных нет, записываем NaN

print("Категориальные переменные:")
print(categorical_summary)

# 3. Формулировка и проверка статистических гипотез

# Гипотеза 1: Средний результат (points) для гонщиков из разных стран отличается
nationality_1 = df[df['nationality'] == 'British']['points']
nationality_2 = df[df['nationality'] == 'German']['points']
t_stat, p_value = ttest_ind(nationality_1, nationality_2)
print(f"Гипотеза 1: p-value = {p_value}")



df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df_cleaned = df.dropna(subset=['grid', 'position'])
# Проверяем, что данные очищены
print(df_cleaned[['grid', 'position']].info())



# Гипотеза 2: Корреляция между стартовой позицией (grid) и финишной позицией (position)
corr, p_value = pearsonr(df_cleaned['grid'], df_cleaned['position'])
print(f"Гипотеза 2: Коэффициент корреляции = {corr}, p-value = {p_value}")

# 4. Кодирование категориальных переменных
# OneHotEncoding для переменной 'nationality_x'
encoder = OneHotEncoder()
encoded_nationality = encoder.fit_transform(df[['nationality']]).toarray()
df_encoded = pd.concat([df, pd.DataFrame(encoded_nationality, columns=encoder.get_feature_names_out(['nationality']))], axis=1)

# LabelEncoding для переменной 'name'
label_encoder = LabelEncoder()
df['name_encoded'] = label_encoder.fit_transform(df['name'])

# 5. Таблица корреляции
# Выбираем только числовые столбцы для вычисления корреляции
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Проверяем, есть ли числовые столбцы
if not numeric_df.empty:
    # Строим таблицу корреляции
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Таблица корреляции (только числовые столбцы)")
    plt.show()
else:
    print("В данных отсутствуют числовые столбцы для вычисления корреляции.")

# 6. Реализация градиентного спуска
# Подготовка данных
X = df['grid'].fillna(df['grid'].mean()).values
y = df['points'].values

# Нормализация данных
X = (X - X.mean()) / X.std()

# Добавление столбца единиц для intercept
X = np.c_[np.ones(X.shape[0]), X]

# Функция потерь (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Стохастический градиентный спуск
def stochastic_gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Инициализация параметров
theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

# Обычный градиентный спуск
theta_gd, cost_history_gd = gradient_descent(X, y, theta, alpha, iterations)

# Стохастический градиентный спуск
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, theta, alpha, iterations)

# Визуализация результатов
plt.plot(cost_history_gd, label='Градиентный спуск')
plt.plot(cost_history_sgd, label='Стохастический градиентный спуск')
plt.xlabel('Итерации')
plt.ylabel('Функция потерь')
plt.legend()
plt.show()