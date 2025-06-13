# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
data = pd.read_csv('C:/Users/avazb/Desktop/BigData/Lab4/train.csv')
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Проверим доступные столбцы
print("Доступные столбцы в данных:")
print(data.columns.tolist())

# 2. Определим целевую переменную (замените 'target_column' на реальное название)
target_column = 'price'  # Измените на правильное название столбца с целевой переменной
if target_column not in data.columns:
    # Если столбец 'price' отсутствует, возьмем последний числовой столбец как целевую переменную
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    target_column = numeric_cols[-1]
    print(f"\nСтолбец 'price' не найден. Используем '{target_column}' как целевую переменную.")

# 2. Разведочный анализ данных
print("\n2. Разведочный анализ данных:")
print(f"a. Размер датафрейма: {data.shape[0]} строк, {data.shape[1]} столбцов")
print(f"b. Объем памяти: {data.memory_usage().sum() / 1024**2:.2f} MB")

# Анализ числовых переменных
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

numeric_cols.remove(target_column)  # Исключаем целевую переменную
print("\nc. Описательные статистики для числовых переменных:")
print(data[numeric_cols].describe(percentiles=[0.25, 0.75]))

# Анализ категориальных переменных
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
print("\nd. Анализ категориальных переменных:")
for col in cat_cols:
    mode_val = data[col].mode()[0]
    mode_count = data[col].value_counts().iloc[0]
    print(f"{col}: мода = {mode_val} (встречается {mode_count} раз)")

# 3. Подготовка данных
# a. Обработка пропусков
print("\n3a. Анализ пропущенных значений:")
print(data.isnull().sum())




# b. Обработка выбросов (визуализация)
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[numeric_cols])
plt.xticks(rotation=45)
plt.title("Распределение числовых переменных")
plt.show()

# c. Кодирование категориальных переменных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# d. Проверка гипотез
print("\n3d. Проверка гипотез:")
# Гипотеза 1: Целевая переменная зависит от первого числового признака
if len(numeric_cols) > 0:
    corr = data[numeric_cols[0]].corr(data[target_column])
    print(f"Корреляция между {numeric_cols[0]} и {target_column}: {corr:.3f}")

# Гипотеза 2: Влияние категориальной переменной (если есть)
if len(cat_cols) > 0:
    cat_price = data.groupby(cat_cols[0])[target_column].mean()
    print(f"\nСреднее значение {target_column} по категориям {cat_cols[0]}:")
    print(cat_price)

# e. Разделение данных
X = data.drop(target_column, axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Построение моделей
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Создание пайплайна с предобработкой и моделью
results = {}
for name, model in models.items():
    try:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # 5. Оценка качества
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\nМодель: {name}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}")
    except Exception as e:
        print(f"\nОшибка при обучении модели {name}: {str(e)}")

# Сравнение моделей
if results:
    results_df = pd.DataFrame(results).T
    print("\nСравнение моделей:")
    print(results_df)

    # Визуализация результатов
    results_df[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 6))
    plt.title('Сравнение метрик ошибок')
    plt.ylabel('Значение метрики')
    plt.xticks(rotation=45)
    plt.show()

    # 6. Сохранение лучшей модели
    best_model_name = results_df['R2'].idxmax()
    print(f"\nЛучшая модель: {best_model_name}")

    for name, model in models.items():
        if name == best_model_name:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, 'best_model.joblib')
            print("Модель сохранена как 'best_model.joblib'")

    # Загрузка модели для проверки
    try:
        loaded_model = joblib.load('best_model.joblib')
        y_pred_loaded = loaded_model.predict(X_test.head())
        print("\nПрогнозы загруженной модели на первых 5 примерах:")
        print(y_pred_loaded)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
else:
    print("Ни одна из моделей не была успешно обучена.")
