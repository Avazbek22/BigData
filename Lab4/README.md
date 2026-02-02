# Лабораторная работа 4

**Тема:** Машинное обучение с учителем. Методы регрессии.

**Сложность:** Medium (общая часть + самостоятельная часть, 2+ модели, сохранение модели).

## Общая часть (Kaggle dataset)
- Датасет: `train.csv` (Paris housing)
- Целевая переменная: `price`
- EDA: размеры, память, статистика числовых, мода категориальных
- Обработка пропусков и выбросов
- 2 гипотезы (корреляция, t-test)
- Модели: LinearRegression, Ridge, KNN, Lasso
- Метрики: MAE, MSE, RMSE, MAPE, R2
- Сохранение лучшей модели в `best_model.joblib`

## Самостоятельная часть (F1 SQLite из Лабы 1)
- Целевая переменная: `points`
- EDA, пропуски, выбросы, гипотезы
- Модели: LinearRegression, Ridge, KNN
- Метрики: MAE, MSE, RMSE, MAPE, R2

## Запуск
```bash
py -3 Lab4\Lab4\Lab4.py
```

## Зависимости
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```
