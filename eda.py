# %%
import os
import keras
import matplotlib.style as style
import numpy as np
import tables
from PIL import Image
from keras import optimizers
from keras.applications import vgg16
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

style.use('seaborn-whitegrid')

# %%
"""
### Open images and convert to NumPy arrays
"""

# %%
pic = Image.open('data/gestures_data/00/01_palm/frame_00_01_0001.png')

# %%
pic

# %%
image = image_utils.load_img(path='data/frame_02.png', target_size=(224, 224))
image = image_utils.img_to_array(image)

# %%
image.shape

# %%
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('data/gestures_data/00'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

# %%
lookup

# %%
def get_data(start, stop):
    x_data = []
    y_data = []
    datacount = 0 # We'll use this to tally how many images are in our dataset
    for i in range(start, stop): # Loop over the ten top-level folders
        for j in os.listdir('./data/gestures_data/0' + str(i) + '/'):
            if not j.startswith('.'): # Again avoid hidden folders
                count = 0 # To tally images of a given gesture
                for k in os.listdir('./data/gestures_data/0' +
                                    str(i) + '/' + j + '/'):
                                    # Loop over the images
                    img = Image.open('./data/gestures_data/0' +
                                     str(i) + '/' + j + '/' + k).convert('L')
                                    # Read in and convert to greyscale
                    img = img.resize((224, 224))
                    arr = np.array(img)
                    x_data.append(arr)
                    count = count + 1
                y_values = np.full((count, 1), lookup[j])
                y_data.append(y_values)
                datacount = datacount + count

    return x_data, y_data

# %%
def process_data(x_data, y_data):
    x_data = np.array(x_data, dtype = 'float32')
    # x_data = np.array(x_data, dtype=np.uint8)
    x_data = x_data.reshape((len(x_data), 224, 224, 1))
    x_data /= 255

    y_data = np.array(y_data)
    y_data = y_data.reshape(len(x_data), 1)
    y_data = to_categorical(y_data)
    return x_data, y_data

# %%
"""
### Train-Test Split
"""

# %%
"""
Train-test split - totally separating images from the first 8 people, and the last 2 people
"""

# %%
X_train, y_train = get_data(0,8)
X_train, y_train = process_data(X_train, y_train)
X_test, y_test = get_data(8,10)
X_test, y_test = process_data(X_test, y_test)

# %%
x_data_2 = []
y_data_2 = []
datacount = 0
for i in range(8,10):
    for j in os.listdir('./data/gestures_data/0' + str(i) + '/'):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir('./data/gestures_data/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('./data/gestures_data/0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((224, 224))
                arr = np.array(img)
                x_data_2.append(arr)
                count = count + 1
            y_values = np.full((count, 1), lookup[j])
            y_data_2.append(y_values)
            datacount = datacount + count
            print(datacount)

# %%
x_data = np.array(x_data_2, dtype = 'float32')
y_data = np.array(y_data_2, dtype = 'float32')

# x_data = x_data.reshape((16000, 224, 224, 1))
# x_data /= 255

# %%
"""
### Save (and load) the X and y
"""

# %%
# hdf5_file = tables.open_file('train_test_split.h5', mode='w')
# X_train = np.array(hdf5_file.root.X_train)
# y_train = np.array(hdf5_file.root.y_train)
# X_test = np.array(hdf5_file.root.X_test)
# y_test = np.array(hdf5_file.root.y_test)
# hdf5_file.close()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=33)

# %%
"""
### VGG Models
"""

# %%
# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(
    weights='imagenet',
    include_top=False,
)
model_vgg16_conv.summary()

# Create your own input format (here 224x224x3)
img_input = Input(shape=(224, 224, 3), name='image_input')

# makes the layers non-trainable
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Use the generated model
output_vgg16_conv = model_vgg16_conv(img_input)

# Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
# x = Dense(4096, activation='relu', name='fc1')(x)
# x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)  # here the 2 indicates binary (3 or more is multiclass)

# Create your own model
my_model = Model(input=img_input, output=x)

# In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


# %%
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
optimizer1 = optimizers.Adam() # Adam is like a gradient descent (way to find parameters)

# %%
class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


# %%
base_model = model1  # Topless
# Add top layer
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)

predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Train top layer
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()

model.fit(X_train[:500], y_train[:500], epochs=4, batch_size=64, validation_data=(X_test[:500], y_test[:500]),
          verbose=1, callbacks=[MetricsCheckpoint('logs')])

# %%
from sklearn.utils import class_weight
class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# %%
x_data.shape

# %%
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
datagen.fit(x_data)
X_batch = datagen.flow(x_data, y_data, batch_size=64)

# %%
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(x_data, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    './data/gestures_data',
    class_mode=None,
    seed=seed)

# mask_generator = mask_datagen.flow_from_directory(
#     'data/masks',
#     class_mode=None,
#     seed=seed)

# combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=200, #originally 2000
    epochs=1) #originally 50

# %%
from keras import models, layers

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dropout(0.25, seed=21))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# %%
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# model.fit(x_data, y_data, epochs=1, batch_size=64, verbose=1)
# model.fit_generator(datagen, samples_per_epoch=len(x_data), epochs=100)
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(X_test, y_test))

# %%
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=len(X_train) / 128, epochs=10, validation_data=(X_test, y_test))


# %%
x_data.shape

# %%
model.summary()

# %%
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))

# %%
"""
### Save/reload the model
"""

# %%
# model.save('my_model_vgg.h5')
model.save('my_model_with_augmentation.h5')

# %%
from keras.models import load_model
model = load_model('my_model_with_augmentation.h5')

# %%
x_data_2 = []
y_data_2 = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(8, 10): # Loop over the ten top-level folders
    for j in os.listdir('/home/ubuntu/project_kojak/data/gestures_data/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('/home/ubuntu/project_kojak/data/gestures_data/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                path = '/home/ubuntu/project_kojak/data/gestures_data/0' + str(i) + '/' + j + '/' + k
                img = image_utils.load_img(path=path, target_size=(224, 224))
                img = image_utils.img_to_array(img)
                arr = np.array(img)
                x_data_2.append(arr)
                count = count + 1
            y_values = np.full((count, 1), lookup[j])
            y_data_2.append(y_values)
            datacount = datacount + count
            print(datacount)

# %%
y_data_2 = np.array(y_data_2)
y_data_2 = y_data_2.reshape(datacount, 1)

from keras.utils import to_categorical
y_data_2 = to_categorical(y_data_2)

# %%
# x_data = np.array(x_data, dtype = 'float32')
x_data_2 = np.array(x_data_2, dtype=np.uint8)
# x_data = x_data.reshape((16000, 224, 224, 1))
# x_data /= 255

# %%
pred = model.predict(x_data_2)

# %%
pred.shape

# %%
predictions = model.predict_classes(x_data)
# predictions = model.predict(x_data_2)

# %%
gesture_names = {0: 'thumb',
                 1: 'palm_moved',
                 2: 'l',
                 3: 'palm',
                 4: 'ok',
                 5: 'down',
                 6: 'index',
                 7: 'c',
                 8: 'fist',
                 9: 'fist_moved'}

# %%
correct = 0
incorrect = 0
for each in pred[0:200]:
    if gesture_names[each] == 'index':
        correct +=1
    else:
        incorrect +=1
print(correct, incorrect)

# %%
from sklearn.metrics import confusion_matrix, classification_report

# %%
"""
### Model #2 Classification Metrics
"""

# %%
confusion_matrix(y_data, predictions)

# %%
print(classification_report(y_data, predictions))

# %%
"""
### Model w/ VGG Classification Metrics
"""

# %%
import defaultdict

for prediction in pred[:100]:
#     print(prediction)
    print(np.where(prediction == prediction.max())[0][0])

# %%
#np.where( == prediction.max())[0][0]

# %%
correct = 0
incorrect = 0
for i in range(len(pred)):
    if (np.where(pred[i] == pred[i].max())[0][0]) == (np.where(y_data_2[i] == y_data_2[i].max())[0][0]):
        correct +=1
    else:
        incorrect +=1
print(correct, incorrect)

# %%
"""
#### Function to get classification metrics
"""

# %%
def get_classification_reports(y_pred, y_true):
    y_pred_classes = np.array(np.argmax(y_pred))  # reconverts back from one hot encoded
    y_true = np.array(np.argmax(y_true))  # reconverts back from one hot encoded
    print(confusion_matrix(y_true, y_pred_classes))
    print(classification_report(y_true, y_pred_classes))

# %%
pred = pred.reshape(len(pred),)

# %%
y_pred_classes = np.argmax(pred, axis = 1)  # reconverts back from one hot encoded
y_true = np.argmax(y_data_2, axis = 1)  # reconverts back from one hot encoded

# %%
confusion_matrix(y_true, y_pred_classes)

# %%
print(classification_report(y_true, y_pred_classes))

# %%
y_true = np.argmax(y_test, axis=1)

# %%
"""
### Model with Image Augmentation
"""

# %%
print(confusion_matrix(y_true, pred))
print('\n')
print(classification_report(y_true, pred))

# %%
for i, v in enumerate(np.bincount(pred)):
    print(i, v)

# %%
for i, v in enumerate(np.bincount(y_true)):
    print(i,v)

# %%
# def process_image(data):
#     x_data = np.array(data, dtype = 'float32')
#     # x_data = np.array(x_data, dtype=np.uint8)
#     x_data = x_data.reshape((len(x_data), 224, 224, 1))
#     x_data /= 255
#     return x_data

def get_prediction_from_image(path):
    x_data = []
    path = path

    img = Image.open(path).convert('L')
    img = img.resize((224, 224))
    arr = np.array(img)
    x_data.append(arr)

    x_data = np.array(x_data, dtype = 'float32')
    x_data = x_data.reshape((len(x_data), 224, 224, 1))
    x_data /= 255
    return gesture_names[model.predict_classes(x_data)[0]]

# %%
get_prediction_from_image('./data/gestures_data/00/01_palm/frame_00_01_0002.png')

# %%
gesture_names = {0: 'thumb',
                 1: 'palm_moved',
                 2: 'l',
                 3: 'palm',
                 4: 'ok',
                 5: 'down',
                 6: 'index',
                 7: 'c',
                 8: 'fist',
                 9: 'fist_moved'}
