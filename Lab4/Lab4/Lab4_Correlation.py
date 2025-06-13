# -*- coding: cp1251 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# �������� ������
file_path = r"C:\Users\avazb\Desktop\BigData\Lab4\train.csv"
data = pd.read_csv(file_path)

if 'id' in data.columns:
    data = data.drop(columns=['id'])

# 1. ������������� ������
# �������� ���������� ������� (���� ����)
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# 2. ������ ������� ����������
corr_matrix = numeric_data.corr()

# 3. ������������
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm',
            center=0,
            linewidths=0.5)
plt.title("������� ���������� ���������")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 4. ���������� ����������
output_path = r"C:\Users\avazb\Desktop\BigData\Lab4\correlation_matrix.png"
plt.savefig(output_path, dpi=300)
plt.show()