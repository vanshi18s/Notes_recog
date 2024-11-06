import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
import numpy as np

data = np.load("/kaggle/input/mnist-data/mnist.npz")
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']

x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32')/255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32')/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', 
input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
loss='categorical_crossentropy',metrics=['accuracy'])

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', 
input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=2, batch_size=64, 
validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

def preprocess_image(image_path):
    
    img = Image.open(image_path).convert('L')  
    img = img.resize((28, 28))  
    img = np.array(img)  
    img = img.astype('float32') / 255  
    img = img.reshape((1, 28, 28, 1))  
    return img

def prediction_made(image_path):
    
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predict_notes = np.argmax(predictions)
    print(f'The predicted alphanumeric character is: {predict_notes}')

image_path = '/kaggle/input/digit-5-for-mnist/IMG_0400.png'
prediction_made(image_path)