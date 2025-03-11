#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:03:50 2025

@author: namin
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 파일 경로 확인
file_path = '/Users/namin/Desktop/대학/3학년 1학기/인공지능개론/AI-Class/AI-Class/Quiz/mobile.csv'
print(f"파일 저장 경로: {os.path.abspath(file_path)}")

# 데이터 읽기
df = pd.read_csv(file_path)

# 특성과 라벨 분리
X = df.drop(columns=["price_range"])  # 특성 (feature)
y = df["price_range"]  # 라벨 (target)

# 데이터 정규화 (SVM과 Logistic Regression에서 성능 향상을 위해)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Decision Tree 분류
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree 정확도: {dt_accuracy:.4f}")

# Random Forest 분류
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest 정확도: {rf_accuracy:.4f}")

# SVM 분류
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM 정확도: {svm_accuracy:.4f}")

# Logistic Regression 분류
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression 정확도: {lr_accuracy:.4f}")
