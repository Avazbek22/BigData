
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv(r"C:\Users\avazb\Desktop\BigData\Lab4\train.csv")


print("Форма датафрейма:", df.shape)

print("Объем памяти:", df.memory_usage(deep=True).sum() / 1024**2, "MB")

# c. Статистика по числовым переменным
print(df.describe(percentiles=[.25, .5, .75]))

# d. Мода для категориальных переменных
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    mode = df[col].mode()[0]
    count = df[col].value_counts()[mode]
    print(f"Мода переменной {col}: {mode}, встречается {count} раз")

# Проверка на пропуски
print("Пропуски")
print(df.isnull().sum())

# Обработка выбросов (пример: удалим значения с площадью > 99 перцентиля)
df = df[df['squareMeters'] < df['squareMeters'].quantile(0.99)]




# 1. Средняя цена домов с бассейном и без
print(ttest_ind(df[df['hasPool'] == 1]['price'], df[df['hasPool'] == 0]['price']))

# 2. Средняя цена по новостройкам и вторичке
print(ttest_ind(df[df['isNewBuilt'] == 1]['price'], df[df['isNewBuilt'] == 0]['price']))

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['id', 'price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование + модель
knn_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

lasso_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

knn_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)



def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

knn_metrics = evaluate(knn_model, X_test, y_test)
lasso_metrics = evaluate(lasso_model, X_test, y_test)

print("🔹 KNN:")
print(knn_metrics)

print("🔹 LASSO:")
print(lasso_metrics)


best_model = lasso_model if lasso_metrics['R2'] > knn_metrics['R2'] else knn_model

if lasso_metrics['R2'] > knn_metrics['R2']:
    print("lasso")
else:
    print("KNN")


print(lasso_metrics['R2'])


joblib.dump(best_model, "best_model.pkl")


# Загрузка модели
try:
    model = joblib.load(r"C:\Users\avazb\Desktop\BigData\Lab4\Lab4\best_model.joblib")
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")



import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(y_test, knn_model.predict(X_test), alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Фактическая цена')
plt.ylabel('Предсказанная цена')
plt.title('KNN: Фактические vs Предсказанные значения')
plt.show()


plt.figure(figsize=(10,6))
plt.scatter(y_test, lasso_model.predict(X_test), alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Фактическая цена')
plt.ylabel('Предсказанная цена')
plt.title('Lasso: Фактические vs Предсказанные значения')
plt.show()