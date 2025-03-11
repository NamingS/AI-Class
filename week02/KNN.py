import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 특성 (꽃잎 길이, 너비 등)
y = iris.target  # 클래스 (0: Setosa, 1: Versicolor, 2: Virginica)

# 2. 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 데이터 정규화 (KNN은 거리 기반 알고리즘이므로 정규화 필요)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. KNN 모델 생성 (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 5. 예측 및 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'KNN 분류 정확도: {accuracy:.2f}')
