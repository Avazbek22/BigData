# -*- coding: cp1251 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
file_path = r"C:\Users\avazb\Desktop\BigData\Lab4\train.csv"
data = pd.read_csv(file_path)

if 'id' in data.columns:
    data = data.drop(columns=['id'])

# 1. Предобработка данных
# Удаление нечисловых колонок (если есть)
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# 2. Расчет матрицы корреляции
corr_matrix = numeric_data.corr()

# 3. Визуализация
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm',
            center=0,
            linewidths=0.5)
plt.title("Матрица корреляции признаков")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 4. Сохранение результата
output_path = r"C:\Users\avazb\Desktop\BigData\Lab4\correlation_matrix.png"
plt.savefig(output_path, dpi=300)
plt.show()