import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os

# Check if model exists
if os.path.exists('mnist_model.h5'):
    print("Loading saved enhanced model...")
    model = load_model('mnist_model.h5')
    plot_history = False
else:
    print("Training new enhanced model...")
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Enhanced preprocessing
    def preprocess_data(x, y):
        x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y = to_categorical(y, 10)
        return x, y

    x_train, y_cat_train = preprocess_data(x_train, y_train)
    x_test, y_cat_test = preprocess_data(x_test, y_test)

    # Enhanced model architecture
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Enhanced training configuration
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    # Train the model
    history = model.fit(x_train, y_cat_train,
                       epochs=15,
                       batch_size=128,
                       validation_data=(x_test, y_cat_test),
                       callbacks=callbacks)

    # Save the model
    model.save('mnist_model.h5')
    print("Enhanced model saved as enhanced_model.h5")
    plot_history = True

# Evaluation and visualization
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_cat_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

if plot_history:
    losses = pd.DataFrame(history.history)
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    losses[['loss','val_loss']].plot()
    plt.title('Loss Curves')

    plt.subplot(1,2,2)
    losses[['accuracy','val_accuracy']].plot()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.show()

# Predictions and metrics
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_cat_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Enhanced digit analysis
def analyze_digit(digit):
    indices = np.where(y_true == digit)[0]
    if len(indices) > 0:
        correct = sum(y_pred_classes[indices] == digit)
        acc = correct/len(indices)
        print(f"\nDigit {digit} Analysis:")
        print(f"Accuracy: {acc:.2%}")
        print(f"Most common error: {np.bincount(y_pred_classes[indices]).argmax() if acc < 1.0 else 'None'}")

        # Show examples
        plt.figure(figsize=(12,3))
        for i in range(min(5, len(indices))):
            plt.subplot(1,5,i+1)
            plt.imshow(x_test[indices[i]].reshape(28,28), cmap='gray')
            pred = y_pred_classes[indices[i]]
            color = 'green' if pred == digit else 'red'
            plt.title(f"Pred: {pred}", color=color)
            plt.axis('off')
        plt.suptitle(f"Sample Digit {digit} Predictions")
        plt.show()
    else:
        print(f"No examples of digit {digit} found")

# Analyze digits 8 and commonly confused digits
for digit in [8, 3, 5, 9]:
    analyze_digit(digit)
