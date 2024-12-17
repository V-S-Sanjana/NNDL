import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
def plot_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()
def load_data(dataset_name):
    if dataset_name == "diabetes":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                   'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, header=None, names=columns)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    elif dataset_name == "cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = pd.DataFrame(data.data, columns=data.feature_names), pd.Series(data.target)
    elif dataset_name == "sonar":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
        df = pd.read_csv(url, header=None)
        X, y = df.iloc[:, :-1], pd.Series(df.iloc[:, -1].apply(lambda x: 1 if x == 'R' else 0))
    return train_test_split(X, y, test_size=0.2, random_state=42)
def train_model(X_train, X_test, y_train, y_test, title):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train.shape[1]),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
    plot_history(history, title)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'{title} Test Accuracy: {test_acc:.2f}')
for name in ["diabetes", "cancer", "sonar"]:
    X_train, X_test, y_train, y_test = load_data(name)
    train_model(X_train, X_test, y_train, y_test, name.capitalize())
