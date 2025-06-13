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

# 1. �������� ������
data = pd.read_csv('C:/Users/avazb/Desktop/BigData/Lab4/train.csv')
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# �������� ��������� �������
print("��������� ������� � ������:")
print(data.columns.tolist())

# 2. ��������� ������� ���������� (�������� 'target_column' �� �������� ��������)
target_column = 'price'  # �������� �� ���������� �������� ������� � ������� ����������
if target_column not in data.columns:
    # ���� ������� 'price' �����������, ������� ��������� �������� ������� ��� ������� ����������
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    target_column = numeric_cols[-1]
    print(f"\n������� 'price' �� ������. ���������� '{target_column}' ��� ������� ����������.")

# 2. ����������� ������ ������
print("\n2. ����������� ������ ������:")
print(f"a. ������ ����������: {data.shape[0]} �����, {data.shape[1]} ��������")
print(f"b. ����� ������: {data.memory_usage().sum() / 1024**2:.2f} MB")

# ������ �������� ����������
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

numeric_cols.remove(target_column)  # ��������� ������� ����������
print("\nc. ������������ ���������� ��� �������� ����������:")
print(data[numeric_cols].describe(percentiles=[0.25, 0.75]))

# ������ �������������� ����������
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
print("\nd. ������ �������������� ����������:")
for col in cat_cols:
    mode_val = data[col].mode()[0]
    mode_count = data[col].value_counts().iloc[0]
    print(f"{col}: ���� = {mode_val} (����������� {mode_count} ���)")

# 3. ���������� ������
# a. ��������� ���������
print("\n3a. ������ ����������� ��������:")
print(data.isnull().sum())




# b. ��������� �������� (������������)
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[numeric_cols])
plt.xticks(rotation=45)
plt.title("������������� �������� ����������")
plt.show()

# c. ����������� �������������� ����������
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# d. �������� �������
print("\n3d. �������� �������:")
# �������� 1: ������� ���������� ������� �� ������� ��������� ��������
if len(numeric_cols) > 0:
    corr = data[numeric_cols[0]].corr(data[target_column])
    print(f"���������� ����� {numeric_cols[0]} � {target_column}: {corr:.3f}")

# �������� 2: ������� �������������� ���������� (���� ����)
if len(cat_cols) > 0:
    cat_price = data.groupby(cat_cols[0])[target_column].mean()
    print(f"\n������� �������� {target_column} �� ���������� {cat_cols[0]}:")
    print(cat_price)

# e. ���������� ������
X = data.drop(target_column, axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ���������� �������
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# �������� ��������� � �������������� � �������
results = {}
for name, model in models.items():
    try:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # 5. ������ ��������
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
        
        print(f"\n������: {name}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}")
    except Exception as e:
        print(f"\n������ ��� �������� ������ {name}: {str(e)}")

# ��������� �������
if results:
    results_df = pd.DataFrame(results).T
    print("\n��������� �������:")
    print(results_df)

    # ������������ �����������
    results_df[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 6))
    plt.title('��������� ������ ������')
    plt.ylabel('�������� �������')
    plt.xticks(rotation=45)
    plt.show()

    # 6. ���������� ������ ������
    best_model_name = results_df['R2'].idxmax()
    print(f"\n������ ������: {best_model_name}")

    for name, model in models.items():
        if name == best_model_name:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, 'best_model.joblib')
            print("������ ��������� ��� 'best_model.joblib'")

    # �������� ������ ��� ��������
    try:
        loaded_model = joblib.load('best_model.joblib')
        y_pred_loaded = loaded_model.predict(X_test.head())
        print("\n�������� ����������� ������ �� ������ 5 ��������:")
        print(y_pred_loaded)
    except Exception as e:
        print(f"������ ��� �������� ������: {str(e)}")
else:
    print("�� ���� �� ������� �� ���� ������� �������.")
