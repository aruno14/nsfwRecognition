import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from collections import Counter
import matplotlib.pyplot as plt
import os

image_size = (48, 48)
batch_size = 32
epochs = 8
model_name = "nsfw_model_gray"

data_folder="content"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
train_generator = datagen.flow_from_directory(
    data_folder,
    target_size=image_size,
    color_mode="grayscale",
    shuffle=True,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    data_folder,
    target_size=image_size,
    color_mode="grayscale",
    shuffle=True,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical')


counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)
print(train_generator.class_indices)
if os.path.exists(model_name):
    print("Load: " + model_name)
    classifier = load_model(model_name)
else:
    classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, input_tensor=None, input_shape=image_size + (1,), pooling=None, classes=2)
    classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    
history = classifier.fit(train_generator, epochs=epochs, validation_data=validation_generator, class_weight=class_weights)
classifier.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-gender-grayscale.png")
plt.show()
plt.close()

