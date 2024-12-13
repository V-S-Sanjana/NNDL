import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes, load_breast_cancer
import pandas as pd

# Load and preprocess datasets
def load_and_preprocess():
    diabetes_data = load_diabetes()
    cancer_data = load_breast_cancer()
    sonar_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data', header=None)

    X_d, y_d = diabetes_data.data, diabetes_data.target
    X_c, y_c = cancer_data.data, cancer_data.target
    X_s, y_s = sonar_data.iloc[:, :-1].values, sonar_data.iloc[:, -1].apply(lambda x: 1 if x == 'M' else 0).values

    scaler = StandardScaler()
    X_d, X_c, X_s = scaler.fit_transform(X_d), scaler.fit_transform(X_c), scaler.fit_transform(X_s)
    
    return train_test_split(X_d, y_d, test_size=0.2, random_state=42), \
           train_test_split(X_c, y_c, test_size=0.2, random_state=42), \
           train_test_split(X_s, y_s, test_size=0.2, random_state=42)

(X_train_d, X_test_d, y_train_d, y_test_d), \
(X_train_c, X_test_c, y_train_c, y_test_c), \
(X_train_s, X_test_s, y_train_s, y_test_s) = load_and_preprocess()

# Build and compile models
def build_model(input_shape, task="classification"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid' if task == "classification" else None)
    ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy' if task == "classification" else 'mse', 
                  metrics=['accuracy'] if task == "classification" else ['mae'])
    return model

# Train and evaluate models
nn_diabetes = build_model((X_train_d.shape[1],), task="regression")
nn_cancer = build_model((X_train_c.shape[1],))
nn_sonar = build_model((X_train_s.shape[1],))

print("Training Diabetes Model...")
nn_diabetes.fit(X_train_d, y_train_d, epochs=50, batch_size=32, validation_data=(X_test_d, y_test_d))
print("Training Cancer Model...")
nn_cancer.fit(X_train_c, y_train_c, epochs=50, batch_size=32, validation_data=(X_test_c, y_test_c))
print("Training Sonar Model...")
nn_sonar.fit(X_train_s, y_train_s, epochs=50, batch_size=32, validation_data=(X_test_s, y_test_s))

# Evaluation
print("\nDiabetes Model MAE:", nn_diabetes.evaluate(X_test_d, y_test_d)[1])
print("Cancer Model Accuracy:", nn_cancer.evaluate(X_test_c, y_test_c)[1] * 100)
print("Sonar Model Accuracy:", nn_sonar.evaluate(X_test_s, y_test_s)[1] * 100)
