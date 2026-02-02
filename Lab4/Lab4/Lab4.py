# -*- coding: utf-8 -*-
"""
Лабораторная работа 4
Машинное обучение с учителем. Методы регрессии
"""

import sqlite3  # работа с SQLite
from pathlib import Path  # удобные пути

import numpy as np  # численные расчеты
import pandas as pd  # табличные данные
import matplotlib  # базовый matplotlib
matplotlib.use("TkAgg")  # отдельные окна графиков
import matplotlib.pyplot as plt  # графики

from scipy.stats import ttest_ind, pearsonr  # статистические тесты
from sklearn.model_selection import train_test_split  # разбиение train/test
from sklearn.preprocessing import StandardScaler  # нормализация
from sklearn.compose import ColumnTransformer  # разные преобразования для разных типов
from sklearn.preprocessing import OneHotEncoder  # one-hot кодирование
from sklearn.pipeline import Pipeline  # пайплайны
from sklearn.neighbors import KNeighborsRegressor  # KNN регрессия
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # линейные модели
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # метрики
import joblib  # сохранение моделей


# -----------------------------
# Вспомогательные функции
# -----------------------------

def basic_info(df: pd.DataFrame, name: str) -> None:
    """Базовая информация о датасете."""
    rows, cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"\n{name}: количество строк = {rows}, количество столбцов = {cols}")
    print(f"{name}: память = {memory_usage} байт")


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика для числовых признаков."""
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    stats = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["min", "50%", "mean", "max", "25%", "75%"]]
    stats.columns = ["min", "median", "mean", "max", "p25", "p75"]
    return stats


def categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика для категориальных признаков."""
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    summary = pd.DataFrame(index=cat_cols, columns=["mode", "mode_count"])
    for col in cat_cols:
        if df[col].notna().any():
            mode_val = df[col].mode()[0]
            mode_count = df[col].value_counts().iloc[0]
            summary.loc[col] = [mode_val, mode_count]
    return summary


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Обработка пропусков: числовые -> медиана, категориальные -> мода."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("unknown")
    return df


def handle_outliers_iqr(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Обработка выбросов по IQR: ограничение значений в [Q1-1.5*IQR, Q3+1.5*IQR]."""
    df = df.copy()
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    for col in numeric_cols:
        if col in exclude:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Подсчет метрик регрессии."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE: считаем только там, где y_true != 0
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}


def build_preprocessor(X: pd.DataFrame):
    """Построение препроцессора: нормализация числовых и OneHot для категорий."""
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor


def evaluate_models(X_train, X_test, y_train, y_test, models: dict) -> pd.DataFrame:
    """Обучение и оценка нескольких моделей."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        results[name] = metrics
        print(f"\nМодель: {name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    return pd.DataFrame(results).T


# -----------------------------
# Часть 1: Общая часть (Kaggle dataset по варианту)
# Вариант 7 -> Париж. Стоимость жилья (train.csv)
# -----------------------------
print("=== ОБЩАЯ ЧАСТЬ: Kaggle dataset ===")

lab4_dir = Path(__file__).resolve().parents[1]
train_path = lab4_dir.parent / "Lab4" / "train.csv"

if not train_path.exists():
    raise FileNotFoundError(f"Файл train.csv не найден: {train_path}")

kaggle_df = pd.read_csv(train_path)

# Удалим id
if "id" in kaggle_df.columns:
    kaggle_df = kaggle_df.drop(columns=["id"])

# Целевая переменная
target_col = "price"
if target_col not in kaggle_df.columns:
    raise ValueError("Целевой столбец 'price' не найден")

# Базовая информация
basic_info(kaggle_df, "Kaggle")

# EDA
print("\nKaggle: числовая статистика")
print(numeric_stats(kaggle_df))
print("\nKaggle: категориальная статистика")
print(categorical_stats(kaggle_df))

# Пропуски
print("\nKaggle: пропуски")
print(kaggle_df.isna().sum())

# Обработка пропусков и выбросов
kaggle_df = handle_missing(kaggle_df)
kaggle_df = handle_outliers_iqr(kaggle_df, exclude=[target_col])

# Гипотезы
print("\nKaggle: гипотезы")
# Гипотеза 1: есть корреляция между площадью и ценой
corr, p_val = pearsonr(kaggle_df["squareMeters"], kaggle_df[target_col])
print(f"Корреляция squareMeters и price: r={corr:.4f}, p-value={p_val:.4e}")
# Гипотеза 2: средняя цена отличается для домов с бассейном и без
pool_yes = kaggle_df[kaggle_df["hasPool"] == 1][target_col]
pool_no = kaggle_df[kaggle_df["hasPool"] == 0][target_col]
if len(pool_yes) > 0 and len(pool_no) > 0:
    t_stat, p_value = ttest_ind(pool_yes, pool_no, equal_var=False)
    print(f"t-test (hasPool): p-value={p_value:.4e}")

# Разделение на признаки и цель
X = kaggle_df.drop(columns=[target_col])
y = kaggle_df[target_col]

# Препроцессор и модели
preprocessor = build_preprocessor(X)

models = {
    "LinearRegression": Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())]),
    "Ridge": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))]),
    "KNN": Pipeline([("preprocessor", preprocessor), ("model", KNeighborsRegressor(n_neighbors=5))]),
    "Lasso": Pipeline([("preprocessor", preprocessor), ("model", Lasso(alpha=0.001, max_iter=2000))]),
}

# Разделение train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение и оценка
results = evaluate_models(X_train, X_test, y_train, y_test, models)
print("\nСравнение моделей:")
print(results)

# Выбор лучшей модели по R2
best_model_name = results["R2"].idxmax()
print(f"\nЛучшая модель: {best_model_name}")

# Сохранение лучшей модели
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
model_path = lab4_dir / "best_model.joblib"
joblib.dump(best_model, model_path)
print(f"Модель сохранена: {model_path}")

# Загрузка модели и проверка
loaded_model = joblib.load(model_path)
y_pred_loaded = loaded_model.predict(X_test.head())
print("\nПрогнозы загруженной модели (первые 5 строк):")
print(y_pred_loaded)


# -----------------------------
# Часть 2: Самостоятельная часть (данные из Лабы 1 - F1 SQLite)
# -----------------------------
print("\n=== САМОСТОЯТЕЛЬНАЯ ЧАСТЬ: F1 SQLite ===")

f1_db = lab4_dir.parent / "Lab1" / "Formula 1 Race Data" / "Formula1.sqlite"
if not f1_db.exists():
    raise FileNotFoundError(f"База данных не найдена: {f1_db}")

conn = sqlite3.connect(f1_db)
query = """
SELECT
    r.raceId,
    r.year,
    r.round,
    r.name AS race_name,
    d.nationality AS driver_nationality,
    con.name AS constructor_name,
    res.grid,
    res.positionOrder,
    res.points,
    res.laps,
    res.statusId
FROM results res
JOIN races r ON r.raceId = res.raceId
JOIN drivers d ON d.driverId = res.driverId
JOIN constructors con ON con.constructorId = res.constructorId
"""

f1_df = pd.read_sql_query(query, conn)
conn.close()

# Целевая переменная для регрессии: points
f1_target = "points"

# Базовая информация
basic_info(f1_df, "F1")

# EDA
print("\nF1: числовая статистика")
print(numeric_stats(f1_df))
print("\nF1: категориальная статистика")
print(categorical_stats(f1_df))

# Пропуски
print("\nF1: пропуски")
print(f1_df.isna().sum())

# Обработка пропусков и выбросов
f1_df = handle_missing(f1_df)
f1_df = handle_outliers_iqr(f1_df, exclude=[f1_target])

# Гипотезы
print("\nF1: гипотезы")
# Гипотеза 1: есть корреляция grid и points
corr, p_val = pearsonr(f1_df["grid"], f1_df[f1_target])
print(f"Корреляция grid и points: r={corr:.4f}, p-value={p_val:.4e}")
# Гипотеза 2: средние points у British и German различаются
brit = f1_df[f1_df["driver_nationality"] == "British"][f1_target]
germ = f1_df[f1_df["driver_nationality"] == "German"][f1_target]
if len(brit) > 0 and len(germ) > 0:
    t_stat, p_value = ttest_ind(brit, germ, equal_var=False)
    print(f"t-test (British vs German): p-value={p_value:.4e}")

# Разделение на признаки и цель
X2 = f1_df.drop(columns=[f1_target])
y2 = f1_df[f1_target]

# Препроцессор и модели
preprocessor2 = build_preprocessor(X2)

models2 = {
    "LinearRegression": Pipeline([("preprocessor", preprocessor2), ("model", LinearRegression())]),
    "Ridge": Pipeline([("preprocessor", preprocessor2), ("model", Ridge(alpha=1.0))]),
    "KNN": Pipeline([("preprocessor", preprocessor2), ("model", KNeighborsRegressor(n_neighbors=5))]),
}

# Разделение train/test
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# Обучение и оценка
results2 = evaluate_models(X2_train, X2_test, y2_train, y2_test, models2)
print("\nСравнение моделей (F1):")
print(results2)

# Лучший алгоритм по R2
best_model_name2 = results2["R2"].idxmax()
print(f"\nЛучшая модель (F1): {best_model_name2}")


"""
ДЛЯ ОТЧЕТА (подсказки для другой ИИ, удалить перед сдачей):

1) Общая часть (Kaggle, Paris housing):
- Датасет train.csv из Lab4.
- Целевой столбец: price.
- Проведены: строки/столбцы, память, статистика числовых, мода категориальных.
- Пропуски заполнены медианой/модой.
- Выбросы ограничены методом IQR (кроме price).
- Гипотеза 1: корреляция squareMeters и price (Pearson).
- Гипотеза 2: средняя price отличается для hasPool=1/0 (t-test).
- Построены модели: LinearRegression, Ridge, KNN, Lasso.
- Метрики: MAE, MSE, RMSE, MAPE, R2.
- Лучшая модель выбрана по R2, сохранена в best_model.joblib, проверена загрузка.

2) Самостоятельная часть (F1 SQLite):
- Данные объединены из results+races+drivers+constructors.
- Целевой столбец: points (регрессия).
- EDA, пропуски, выбросы.
- Гипотеза 1: корреляция grid и points.
- Гипотеза 2: средние points British vs German.
- Модели: LinearRegression, Ridge, KNN.
- Метрики: MAE, MSE, RMSE, MAPE, R2.

3) Ответы на контрольные вопросы:
- Регрессия: прогноз численного значения.
- Линейная регрессия: зависимость целевой переменной от признаков через линейную комбинацию.
- LASSO и ElasticNet: модели с регуляризацией (L1, L1+L2).
- Метрики: MAE, MSE, RMSE, MAPE, R2.
- Нормализация: приведение в диапазон, стандартизация: (x-mean)/std.
"""
