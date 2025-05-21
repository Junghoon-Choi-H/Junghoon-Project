import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 학습 데이터 생성 (비선형 + 노이즈 포함)
np.random.seed(42)
X = np.linspace(-5, 5, 1000).reshape(-1, 1)  # 더 넓고 촘촘하게
true_y = X**3 - 2 * X**2 + X + 5
noise = np.random.normal(loc=0.0, scale=5.0, size=true_y.shape)  # 노이즈 유지
y = true_y + noise

# 2. 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 3. 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. 모델 학습
model.fit(X, y, epochs=500, verbose=0)

# 5. 예측
new_input = np.array([[2.0]])
prediction = model.predict(new_input)
print(f"{new_input[0][0]}에 대한 예측 결과: {prediction[0][0]:.2f}")

# 6. 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Noisy Data', alpha=0.3, s=10)
plt.plot(X, true_y, label='True Function', linestyle='dashed')
plt.plot(X, model.predict(X), label='Model Prediction', color='red')
plt.legend()
plt.title("Deep Neural Network with Extended Noisy Data")
plt.show()
