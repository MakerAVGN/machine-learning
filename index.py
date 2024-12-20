import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Настройка стиля графиков
sns.set_theme(style="whitegrid")

# Загрузка датасета
data = pd.read_csv("dataset2.csv")  # Укажите правильный путь к вашему файлу

# Просмотр первых строк датасета
print(data.head())

# Проверка структуры данных
print(data.info())

# Удаление дубликатов, если есть
data = data.drop_duplicates()

# Проверка пропусков
if data.isnull().sum().sum() > 0:
    print("Пропуски найдены, заполняем их медианой.")
    data = data.fillna(data.median(numeric_only=True))

# Предположим, что 'activityID' - целевая переменная
X = data.drop(columns=['activityID'])  # Признаки
y = data['activityID']  # Целевая переменная

# Преобразование категориальной целевой переменной в числовой формат
y = pd.factorize(y)[0]  # Преобразуем строковые метки в числа
y = to_categorical(y)   # One-hot encoding

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # Количество категорий активности

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Тестовая точность: {accuracy:.2f}")

# Построение графика обучения
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("График точности обучения")
plt.xlabel("Эпохи")
plt.ylabel("Точность")
plt.legend()
plt.show()

# Анализ данных (пример с пульсом)
if 'heart_rate' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['heart_rate'], bins=30, kde=True, color="blue")
    plt.title("Распределение пульса")
    plt.xlabel("Пульс")
    plt.ylabel("Частота")
    plt.show()

# Анализ данных акселерации
if {'hand acceleration X ±16g', 'hand acceleration Y ±16g', 'hand acceleration Z ±16g'}.issubset(data.columns):
    data['total_acceleration'] = np.sqrt(
        data['hand acceleration X ±16g']**2 +
        data['hand acceleration Y ±16g']**2 +
        data['hand acceleration Z ±16g']**2
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(data['total_acceleration'], bins=30, kde=True, color="purple")
    plt.title("Распределение общей акселерации")
    plt.xlabel("Акселерация")
    plt.ylabel("Частота")
    plt.show()

# Предсказание новой активности
new_sample = np.array(X_test[0]).reshape(1, -1)  # Используем первый образец из теста
predicted_class = model.predict(new_sample).argmax()
print(f"Предсказанная активность: {predicted_class}")
