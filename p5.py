import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load the dataset
# Download dataset from the link, load it into your system and use it.
# The dataset is assumed to be a simple text file, here we'll use 'text_data.txt' as the filename.
# Replace the filename with the correct path once the dataset is downloaded.

# Read the text data
with open("1661-0.txt", "r") as f:
    text = f.read().lower()

# 2. Tokenize the text and create sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])  # Fit the tokenizer on the entire text
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 to account for the padding token

# Convert the text into sequences of integers
sequences = tokenizer.texts_to_sequences([text])[0]

# Create input-output pairs (X, y)
sequence_length = 5  # Length of the sequence to predict the next word

X, y = [], []
for i in range(sequence_length, len(sequences)):
    X.append(sequences[i-sequence_length:i])
    y.append(sequences[i])

X = np.array(X)
y = np.array(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=sequence_length))
model.add(LSTM(128, return_sequences=False))  # LSTM layer with 128 units
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(vocab_size, activation='softmax'))  # Output layer with softmax activation for classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 5. Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 7. Plot training and validation accuracy/loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# 8. Predict the next word
def predict_next_word(model, tokenizer, text, sequence_length):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])[0]
    
    # If the sequence is shorter than the required length, pad it
    if len(sequence) < sequence_length:
        sequence = [0] * (sequence_length - len(sequence)) + sequence
    
    # Predict the next word
    predicted = model.predict(np.array([sequence]))  # Predict the next word
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    
    # Convert the predicted index back to word
    predicted_word = tokenizer.index_word[predicted_word_index]
    
    return predicted_word

# Example of predicting the next word
input_text = "the quick brown fox"  # Example input sequence
predicted_word = predict_next_word(model, tokenizer, input_text, sequence_length)
print(f"Predicted next word after '{input_text}': {predicted_word}")
