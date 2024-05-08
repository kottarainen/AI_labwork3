import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load MNIST dataset and preprocess data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Visualize dataset images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(train_images[i], cmap='gray')
    axs[i].set_title(f'Label: {train_labels[i]}')
plt.show()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# implement image augmentation by creating ImageDataGenerator object and specifying augmentation techniques
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.8, 1.2),    
    fill_mode='nearest'
)

datagen.fit(train_images.reshape(-1, 28, 28, 1))

# visualize training images after augmentation
augmented_images, _ = next(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=5))
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(augmented_images[i].squeeze(), cmap='gray')
    axs[i].set_title(f'Label: {train_labels[i]}')
plt.show()

# define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# define callbacks for saving the model with the lowest validation loss
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras',
                                                         monitor='val_loss',
                                                         save_best_only=True,
                                                         save_weights_only=False,
                                                         mode='min',
                                                         verbose=1)


# Train the model
history = model.fit(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=32),
                    epochs=10,
                    validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels),
                    callbacks=[checkpoint_callback])

# Save the model at the last training epoch
model.save('last_epoch_model.keras')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model performance with the testing dataset
test_loss, test_accuracy = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Display prediction results of the best-performing model
predictions = model.predict(test_images.reshape(-1, 28, 28, 1))
best_prediction_index = np.argmax(predictions[0])
best_prediction_label = np.argmax(predictions[0])
best_prediction_probability = predictions[0][best_prediction_index]

plt.imshow(test_images[0], cmap='gray')
plt.title(f'Predicted: {best_prediction_label}, Probability: {best_prediction_probability:.2f}')
plt.show()