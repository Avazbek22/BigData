# -*- coding: cp1251 -*-

import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ����������� � ���� ������ SQLite
database = r'C:\Users\avazb\Desktop\BigData\Lab1\Formula 1 Race Data\Formula1.sqlite'
conn = sqlite3.connect(database)

# �������� ������ �� ������
races = pd.read_sql_query("SELECT * FROM races", conn)
drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
constructors = pd.read_sql_query("SELECT * FROM constructors", conn)
results = pd.read_sql_query("SELECT * FROM results", conn)
circuits = pd.read_sql_query("SELECT * FROM circuits", conn)

# �������� ���������� � ����� ������
conn.close()

# ���������� ���� ������ ������� 'constructorId' � int64 � ������� constructors
constructors['constructorId'] = constructors['constructorId'].astype('int64')

# �������� ������������� �������� ����� ������������
if 'url' in circuits.columns:
    circuits = circuits.drop(columns=['url'])

# ����������� ������ � ��������� ���������������� ���������
df = pd.merge(results, races, on='raceId', how='left', suffixes=('_results', '_races'))
df = pd.merge(df, drivers, on='driverId', how='left', suffixes=('', '_drivers'))
df = pd.merge(df, constructors, on='constructorId', how='left', suffixes=('', '_constructors'))
df = pd.merge(df, circuits, on='circuitId', how='left', suffixes=('', '_circuits'))

# 1. ���������� ����� � ��������
rows, cols = df.shape
print(f"���������� �����: {rows}, ���������� ��������: {cols}")

# 2. ����������� ������ ������

# (a) ��� �������� ����������
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# ������� ������ DataFrame ��� �������� �����������
numeric_summary = pd.DataFrame(index=numeric_columns, columns=[
    '���� ���������', '�������', '��������', '�������', '�������', '���������', 
    '�������� 0.1', '�������� 0.9', '�������� 1', '�������� 3'
])

# ��������� DataFrame
for col in numeric_columns:
    numeric_summary.loc[col, '���� ���������'] = df[col].isnull().mean()
    numeric_summary.loc[col, '�������'] = df[col].min()
    numeric_summary.loc[col, '��������'] = df[col].max()
    numeric_summary.loc[col, '�������'] = df[col].mean()
    numeric_summary.loc[col, '�������'] = df[col].median()
    numeric_summary.loc[col, '���������'] = df[col].var()
    numeric_summary.loc[col, '�������� 0.1'] = df[col].quantile(0.1)
    numeric_summary.loc[col, '�������� 0.9'] = df[col].quantile(0.9)
    numeric_summary.loc[col, '�������� 1'] = df[col].quantile(0.25)
    numeric_summary.loc[col, '�������� 3'] = df[col].quantile(0.75)

print("�������� ����������:")
print(numeric_summary)

# (b) ��� �������������� ����������
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# ������� ������ DataFrame ��� �������� �����������
categorical_summary = pd.DataFrame(index=categorical_columns, columns=[
    '���� ���������', '���������� ���������� ��������', '����'
])

# ��������� DataFrame
for col in categorical_columns:
    categorical_summary.loc[col, '���� ���������'] = df[col].isnull().mean()
    categorical_summary.loc[col, '���������� ���������� ��������'] = df[col].nunique()
    
    # ���������, ���� �� ������ � �������
    if not df[col].empty and df[col].notna().any():
        categorical_summary.loc[col, '����'] = df[col].mode()[0]  # ����� ������ ����
    else:
        categorical_summary.loc[col, '����'] = np.nan  # ���� ������ ���, ���������� NaN

print("�������������� ����������:")
print(categorical_summary)

# 3. ������������ � �������� �������������� �������

# �������� 1: ������� ��������� (points) ��� �������� �� ������ ����� ����������
nationality_1 = df[df['nationality'] == 'British']['points']
nationality_2 = df[df['nationality'] == 'German']['points']
t_stat, p_value = ttest_ind(nationality_1, nationality_2)
print(f"�������� 1: p-value = {p_value}")



df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df_cleaned = df.dropna(subset=['grid', 'position'])
# ���������, ��� ������ �������
print(df_cleaned[['grid', 'position']].info())



# �������� 2: ���������� ����� ��������� �������� (grid) � �������� �������� (position)
corr, p_value = pearsonr(df_cleaned['grid'], df_cleaned['position'])
print(f"�������� 2: ����������� ���������� = {corr}, p-value = {p_value}")

# 4. ����������� �������������� ����������
# OneHotEncoding ��� ���������� 'nationality_x'
encoder = OneHotEncoder()
encoded_nationality = encoder.fit_transform(df[['nationality']]).toarray()
df_encoded = pd.concat([df, pd.DataFrame(encoded_nationality, columns=encoder.get_feature_names_out(['nationality']))], axis=1)

# LabelEncoding ��� ���������� 'name'
label_encoder = LabelEncoder()
df['name_encoded'] = label_encoder.fit_transform(df['name'])

# 5. ������� ����������
# �������� ������ �������� ������� ��� ���������� ����������
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# ���������, ���� �� �������� �������
if not numeric_df.empty:
    # ������ ������� ����������
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("������� ���������� (������ �������� �������)")
    plt.show()
else:
    print("� ������ ����������� �������� ������� ��� ���������� ����������.")

# 6. ���������� ������������ ������
# ���������� ������
X = df['grid'].fillna(df['grid'].mean()).values
y = df['points'].values

# ������������ ������
X = (X - X.mean()) / X.std()

# ���������� ������� ������ ��� intercept
X = np.c_[np.ones(X.shape[0]), X]

# ������� ������ (MSE)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# ����������� �����
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# �������������� ����������� �����
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

# ������������� ����������
theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

# ������� ����������� �����
theta_gd, cost_history_gd = gradient_descent(X, y, theta, alpha, iterations)

# �������������� ����������� �����
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, theta, alpha, iterations)

# ������������ �����������
plt.plot(cost_history_gd, label='����������� �����')
plt.plot(cost_history_sgd, label='�������������� ����������� �����')
plt.xlabel('��������')
plt.ylabel('������� ������')
plt.legend()
plt.show()