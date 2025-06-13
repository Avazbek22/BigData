# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score



# 1. �������� ������
data = pd.read_csv(r"C:\Users\avazb\Desktop\BigData\Lab5\Iris.csv")
print("������ 5 ����� ������:")
print(data.head())

# 2. EDA � �������������
print("\n���������� � ������:")
print(data.info())

print("\n������������ ����������:")
print(data.describe())

# �������� �� �������
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
plt.figure(figsize=(12, 6))
data[numeric_cols].boxplot()
plt.title("����� � ����� �� ��������� ��������")
plt.show()

# ��������� ��������
for col in numeric_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

plt.figure(figsize=(12, 6))
data[numeric_cols].boxplot()
plt.title("����� � ����� ����� ��������� ��������")
plt.show()

# ������������ ������
scaler = StandardScaler()
X = data[numeric_cols]
y = data['Species']
X_scaled = scaler.fit_transform(X)

# 3. Kernel PCA � ������� ������
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

plt.figure(figsize=(20, 15))
for i, kernel in enumerate(kernels, 1):
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_scaled)
    
    plt.subplot(2, 3, i)
    scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y.astype('category').cat.codes)
    plt.title(f'Kernel PCA ({kernel} kernel)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(y.unique()))

plt.tight_layout()
plt.show()

# 5. ������ ��� ��������� ����
kpca_linear = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
X_kpca_linear = kpca_linear.fit_transform(X_scaled)
X_back = kpca_linear.inverse_transform(X_kpca_linear)

# ���������� lost_variance
lost_variance = np.mean(np.abs(X_scaled - X_back))
print(f"\nLost variance ��� ��������� ����: {lost_variance:.4f}")

# 6. ��������� � t-SNE � UMAP
# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype('category').cat.codes)
plt.title('t-SNE ��������')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.unique()))
plt.show()

# UMAP
umap_model = umap.UMAP(random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y.astype('category').cat.codes)
plt.title('UMAP ��������')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=list(y.unique()))
plt.show()

# 7. ���������� � �������� ������
# ���������� ������
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# �������� ������
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ���������� ������
joblib.dump(model, 'iris_model.joblib')
print("\n������ ��������� � ���� 'iris_model.joblib'")

# �������� ������
loaded_model = joblib.load('C:/Users/avazb/Desktop/BigData/Lab5/Lab5/iris_model.joblib')
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n�������� ����������� ������: {accuracy:.4f}")


# ������������ ��������� K-means � �������������� PCA
from sklearn.decomposition import PCA

# ��������� PCA ��� ������������ � 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ������������� K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# ������������
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='���������')
plt.title('K-means ������������� (k=3) � PCA ���������')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.colorbar(scatter, label='�������')
plt.show()

# ������ �������� �������������
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score ��� K-means: {silhouette:.3f}")

# �������� ������ K-means
joblib.dump(kmeans, 'kmeans_model.joblib')
print("������ K-means ��������� � ���� 'kmeans_model.joblib'")


print("//////////////////////////")

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ������������� �������������
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

# ������������ ������������
plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, orientation='top', truncate_mode='lastp', p=12)
plt.title('������������ ������������� ������������� (Ward linkage)')
plt.xlabel('������ �������')
plt.ylabel('����������')
plt.show()

# ������������ ���������
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='plasma')
plt.title('������������� ������������� (k=3) � PCA ���������')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='�������')
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
true_labels = le.fit_transform(y)

# ������ ��������
silhouette_agg = silhouette_score(X_scaled, agg_labels)
ari_agg = adjusted_rand_score(true_labels, agg_labels)
print(f"Silhouette Score ��� ������������� �������������: {silhouette_agg:.3f}")
print(f"Adjusted Rand Index: {ari_agg:.3f}")

# �������� ������
joblib.dump(agg_clustering, 'agg_clustering.joblib')
print("������ ������������� ������������� ��������� � ���� 'agg_clustering.joblib'")