# -*- coding: utf-8 -*-
"""
Лабораторная работа №2
Исследовательский анализ данных. Постановка гипотез. Категориальные данные
"""

import sqlite3  # работа с SQLite
from pathlib import Path  # удобные пути

import numpy as np  # численные расчеты
import pandas as pd  # табличные данные
import seaborn as sns  # готовые датасеты и визуализация
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # принудительно показывать отдельные окна графиков
import matplotlib.pyplot as plt  # графики
from scipy.stats import ttest_ind, pearsonr  # статистические тесты
from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # кодирование категорий


# -----------------------------
# Вспомогательные функции
# -----------------------------

def numeric_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная статистика для числовых признаков."""
    numeric_columns = df.select_dtypes(include=["float64", "int64", "int32"]).columns
    summary = pd.DataFrame(index=numeric_columns, columns=[
        "Доля пропусков", "Минимум", "Максимум", "Среднее", "Медиана", "Дисперсия",
        "Квантиль 0.1", "Квантиль 0.9", "Квартиль 1", "Квартиль 3"
    ])
    for col in numeric_columns:
        summary.loc[col, "Доля пропусков"] = df[col].isna().mean()
        summary.loc[col, "Минимум"] = df[col].min()
        summary.loc[col, "Максимум"] = df[col].max()
        summary.loc[col, "Среднее"] = df[col].mean()
        summary.loc[col, "Медиана"] = df[col].median()
        summary.loc[col, "Дисперсия"] = df[col].var()
        summary.loc[col, "Квантиль 0.1"] = df[col].quantile(0.1)
        summary.loc[col, "Квантиль 0.9"] = df[col].quantile(0.9)
        summary.loc[col, "Квартиль 1"] = df[col].quantile(0.25)
        summary.loc[col, "Квартиль 3"] = df[col].quantile(0.75)
    return summary


def categorical_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная статистика для категориальных признаков."""
    # Явно выбираем строковые и категориальные признаки (без предупреждений pandas)
    cat_columns = df.select_dtypes(include=["object", "category", "string"]).columns
    summary = pd.DataFrame(index=cat_columns, columns=[
        "Доля пропусков", "Количество уникальных значений", "Мода"
    ])
    for col in cat_columns:
        summary.loc[col, "Доля пропусков"] = df[col].isna().mean()
        summary.loc[col, "Количество уникальных значений"] = df[col].nunique()
        if df[col].notna().any():
            summary.loc[col, "Мода"] = df[col].mode()[0]
        else:
            summary.loc[col, "Мода"] = np.nan
    return summary


def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Функция потерь (MSE)."""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, iterations: int):
    """Обычный градиентный спуск."""
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history


def stochastic_gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, iterations: int):
    """Стохастический градиентный спуск."""
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        for _i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history


# -----------------------------
# Часть 1: Общая часть (dataset mpg)
# -----------------------------
print("=== ОБЩАЯ ЧАСТЬ: dataset mpg ===")

# Загрузка данных mpg
mpg = sns.load_dataset("mpg")  # обязательный датасет по ТЗ

# 1) Количество строк и столбцов
rows, cols = mpg.shape
print(f"mpg: количество строк = {rows}, количество столбцов = {cols}")

# 2) Разведочный анализ
mpg_numeric = numeric_eda(mpg)
mpg_categorical = categorical_eda(mpg)
print("\nmpg: числовые переменные")
print(mpg_numeric)
print("\nmpg: категориальные переменные")
print(mpg_categorical)

# 3) Гипотезы (минимум 2) + обоснование критериев
# Гипотеза 1: Средний mpg различается между машинами из USA и Japan.
# Обоснование: сравнение средних двух независимых выборок -> t-test.
usa = mpg[mpg["origin"] == "usa"]["mpg"].dropna()
japan = mpg[mpg["origin"] == "japan"]["mpg"].dropna()
if len(usa) > 0 and len(japan) > 0:
    t_stat, p_value = ttest_ind(usa, japan, equal_var=False)
    print(f"\nГипотеза 1 (mpg USA vs Japan, t-test): p-value = {p_value}")

# Гипотеза 2: Между weight и mpg есть отрицательная корреляция.
# Обоснование: проверка линейной связи -> коэффициент Пирсона.
mpg_clean = mpg.dropna(subset=["weight", "mpg"])
if len(mpg_clean) > 0:
    corr, p_value = pearsonr(mpg_clean["weight"], mpg_clean["mpg"])
    print(f"Гипотеза 2 (корреляция weight и mpg, Pearson): r = {corr}, p-value = {p_value}")

# 4) Кодирование категориальных переменных
# OneHotEncoding для origin
encoder = OneHotEncoder(sparse_output=False)
origin_ohe = encoder.fit_transform(mpg[["origin"]].dropna())
origin_cols = encoder.get_feature_names_out(["origin"])
mpg_ohe = pd.DataFrame(origin_ohe, columns=origin_cols)

# LabelEncoding для name (как пример)
label_encoder = LabelEncoder()
mpg_name_encoded = label_encoder.fit_transform(mpg["name"].fillna("unknown"))
mpg["name_encoded"] = mpg_name_encoded

# 5) Корреляция признаков и целевого столбца
# Целевой столбец: mpg (экономичность)
# Признаки: числовые столбцы, кроме mpg
numeric_cols = mpg.select_dtypes(include=["float64", "int64", "int32"]).columns
if "mpg" in numeric_cols:
    feature_cols = [c for c in numeric_cols if c != "mpg"]
    corr_with_target = mpg[feature_cols + ["mpg"]].corr()["mpg"].sort_values(ascending=False)
    print("\nКорреляции с целевым столбцом mpg:")
    print(corr_with_target)

# 6) Градиентный спуск (mpg ~ horsepower)
# Подготовка данных
mpg_gd = mpg.dropna(subset=["horsepower", "mpg"])
X = mpg_gd["horsepower"].values
y = mpg_gd["mpg"].values

# Нормализация X
X = (X - X.mean()) / X.std()

# Добавление столбца единиц (bias)
X = np.c_[np.ones(X.shape[0]), X]

# Инициализация
theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 500

# Обычный и стохастический градиентный спуск
theta_gd, cost_gd = gradient_descent(X, y, theta, alpha, iterations)
theta_sgd, cost_sgd = stochastic_gradient_descent(X, y, theta, alpha, iterations)

# График сходимости
plt.figure(figsize=(10, 4))
plt.plot(cost_gd, label="Градиентный спуск")
plt.plot(cost_sgd, label="Стохастический градиентный спуск")
plt.xlabel("Итерации")
plt.ylabel("Функция потерь")
plt.title("Сходимость градиентных спусков (mpg ~ horsepower)")
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# Часть 2: Самостоятельная часть (вариант, БД F1)
# -----------------------------
print("\n=== САМОСТОЯТЕЛЬНАЯ ЧАСТЬ: F1 SQLite ===")

# Путь к БД (относительно проекта)
base_dir = Path(__file__).resolve().parents[1]  # папка Lab2
f1_db = base_dir.parent / "Lab1" / "Formula 1 Race Data" / "Formula1.sqlite"

if not f1_db.exists():
    raise FileNotFoundError(f"База данных не найдена: {f1_db}")

# Подключение к БД
conn = sqlite3.connect(f1_db)

# Загрузка таблиц
races = pd.read_sql_query("SELECT * FROM races", conn)
drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
constructors = pd.read_sql_query("SELECT * FROM constructors", conn)
results = pd.read_sql_query("SELECT * FROM results", conn)
circuits = pd.read_sql_query("SELECT * FROM circuits", conn)

conn.close()

# Приводим ключи к одинаковому типу, чтобы merge работал корректно
results["constructorId"] = pd.to_numeric(results["constructorId"], errors="coerce")
constructors["constructorId"] = pd.to_numeric(constructors["constructorId"], errors="coerce")

# Объединение данных
if "url" in circuits.columns:
    circuits = circuits.drop(columns=["url"])

f1 = pd.merge(results, races, on="raceId", how="left", suffixes=("_results", "_races"))
f1 = pd.merge(f1, drivers, on="driverId", how="left", suffixes=("", "_drivers"))
f1 = pd.merge(f1, constructors, on="constructorId", how="left", suffixes=("", "_constructors"))
f1 = pd.merge(f1, circuits, on="circuitId", how="left", suffixes=("", "_circuits"))

# 1) Количество строк и столбцов
rows, cols = f1.shape
print(f"F1: количество строк = {rows}, количество столбцов = {cols}")

# 2) Разведочный анализ
f1_numeric = numeric_eda(f1)
f1_categorical = categorical_eda(f1)
print("\nF1: числовые переменные")
print(f1_numeric)
print("\nF1: категориальные переменные")
print(f1_categorical)

# 3) Гипотезы + обоснование
# Гипотеза 1: Средние очки (points) различаются у британских и немецких гонщиков.
# Обоснование: сравнение средних двух независимых выборок -> t-test.
brit = f1[f1["nationality"] == "British"]["points"].dropna()
germ = f1[f1["nationality"] == "German"]["points"].dropna()
if len(brit) > 0 and len(germ) > 0:
    t_stat, p_value = ttest_ind(brit, germ, equal_var=False)
    print(f"\nГипотеза 1 (points British vs German, t-test): p-value = {p_value}")

# Гипотеза 2: Есть корреляция между стартовой позицией (grid) и финишной (position).
# Обоснование: проверка линейной связи -> коэффициент Пирсона.
f1["grid"] = pd.to_numeric(f1["grid"], errors="coerce")
f1["position"] = pd.to_numeric(f1["position"], errors="coerce")
f1_clean = f1.dropna(subset=["grid", "position"])
if len(f1_clean) > 0:
    corr, p_value = pearsonr(f1_clean["grid"], f1_clean["position"])
    print(f"Гипотеза 2 (grid vs position, Pearson): r = {corr}, p-value = {p_value}")

# 4) Кодирование категориальных переменных
# OneHotEncoding для nationality
enc = OneHotEncoder(sparse_output=False)
encoded_nat = enc.fit_transform(f1[["nationality"]].fillna("Unknown"))
encoded_nat_df = pd.DataFrame(encoded_nat, columns=enc.get_feature_names_out(["nationality"]))

# LabelEncoding для name (фамилия гонщика)
le = LabelEncoder()
f1["surname_encoded"] = le.fit_transform(f1["surname"].fillna("Unknown"))

# 5) Корреляция признаков и целевого столбца
# Целевой столбец: points (результат гонщика)
# Признаки: числовые столбцы, кроме points
numeric_cols_f1 = f1.select_dtypes(include=["float64", "int64", "int32"]).columns
if "points" in numeric_cols_f1:
    feature_cols_f1 = [c for c in numeric_cols_f1 if c != "points"]
    corr_with_target_f1 = f1[feature_cols_f1 + ["points"]].corr()["points"].sort_values(ascending=False)
    print("\nКорреляции с целевым столбцом points:")
    print(corr_with_target_f1)

# 6) Градиентный спуск (points ~ grid)
# Подготовка данных
f1_gd = f1.dropna(subset=["grid", "points"])
X2 = f1_gd["grid"].values
y2 = f1_gd["points"].values

# Нормализация X
X2 = (X2 - X2.mean()) / X2.std()

# Добавление столбца единиц
X2 = np.c_[np.ones(X2.shape[0]), X2]

# Инициализация
theta2 = np.zeros(X2.shape[1])
alpha2 = 0.01
iterations2 = 300

# Градиентные спуски
theta2_gd, cost2_gd = gradient_descent(X2, y2, theta2, alpha2, iterations2)
theta2_sgd, cost2_sgd = stochastic_gradient_descent(X2, y2, theta2, alpha2, iterations2)

# График сходимости
plt.figure(figsize=(10, 4))
plt.plot(cost2_gd, label="Градиентный спуск")
plt.plot(cost2_sgd, label="Стохастический градиентный спуск")
plt.xlabel("Итерации")
plt.ylabel("Функция потерь")
plt.title("Сходимость градиентных спусков (points ~ grid)")
plt.legend()
plt.tight_layout()
plt.show()


"""
ДЛЯ ОТЧЕТА (подсказки для другой ИИ, удалить перед сдачей):

1) Общая часть (mpg):
- Датасет загружен через sns.load_dataset('mpg').
- Посчитаны строки/столбцы.
- EDA: числовые (доля пропусков, min/max, mean, median, var, q0.1/q0.9, Q1/Q3),
  категориальные (доля пропусков, уникальные, мода).
- Гипотеза 1: средний mpg различается у USA и Japan -> t-test (две независимые выборки).
- Гипотеза 2: корреляция weight и mpg -> Pearson (линейная связь).
- Кодирование: OneHot для origin, Label для name.
- Целевой столбец: mpg, признаки — остальные числовые; корреляция с mpg выведена.
- Градиентный спуск и SGD: y=mpg, x=horsepower, график сходимости.

2) Самостоятельная часть (вариант, F1 SQLite):
- Данные собраны через объединение results+races+drivers+constructors+circuits.
- Посчитаны строки/столбцы.
- EDA по числовым и категориальным признакам.
- Гипотеза 1: средние points у British и German -> t-test.
- Гипотеза 2: корреляция grid и position -> Pearson.
- Кодирование: OneHot для nationality, Label для surname.
- Целевой столбец: points, признаки — остальные числовые; корреляция с points выведена.
- Градиентный спуск и SGD: y=points, x=grid, график сходимости.

3) Обоснование критериев:
- t-test для сравнения средних двух независимых групп.
- Pearson для оценки линейной корреляции числовых признаков.
"""
