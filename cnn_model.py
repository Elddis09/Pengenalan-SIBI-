# Bagian 1 - Membangun CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Inisialisasi CNN
classifier = Sequential()

# Langkah 1 - Layer Konvolusi
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Langkah 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Menambahkan layer konvolusi kedua
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Menambahkan layer konvolusi ketiga
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Langkah 3 - Flattening
classifier.add(Flatten())

# Langkah 4 - Full Connection
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(8, activation='softmax'))

# Mengompilasi CNN
classifier.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Bagian 2 - Memasukkan CNN ke gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Training the model
history = classifier.fit(
    training_set,
    steps_per_epoch=50,
    epochs=100,
    validation_data=test_set,
    validation_steps=100
)

# Menyimpan model
classifier.save('Trained_model.h5')

# Visualisasi hasil pelatihan model
print(history.history.keys())
import matplotlib.pyplot as plt

# Merangkum riwayat untuk akurasi
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi model')
plt.ylabel('Akurasi')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Merangkum riwayat untuk loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
