import os
from os.path import exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2 , preprocess_input as inception_resnet_preprocessor
from keras.applications.xception import Xception , preprocess_input as xception_preprocessor
from keras.applications.vgg16 import VGG16 , preprocess_input as vgg16_preprocessor
from keras.applications.vgg19 import VGG19 , preprocess_input as vgg19_preprocessor
from keras.applications.resnet50 import ResNet50 , preprocess_input as resnet50_preprocessor
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping, Callback

######## Prepare and Preprocess the data ##########################
#Labels.csv file has an id and breed for each image
labels = pd.read_csv('labels.csv')
print(labels.head())

#see the dog breed distribution
plt.figure(figsize=(18, 8))
labels['breed'].value_counts().plot(kind='bar')

#Top 5 breeds
print("Top five breeds:")
print(labels['breed'].value_counts().head())

#Bottom 5 breeds
print()
print("Five rare breeds:")
print(labels['breed'].value_counts().tail())


####### Xception ###################################################
seed = 1986
image_size = 299
train_dir = 'train'
test_dir = 'test'

#encode labels
le = LabelEncoder()
train_labels = le.fit_transform(labels['breed'])
labels['class'] = train_labels
print(labels.head())


#load training data
train = np.array([img_to_array(load_img(join(train_dir, id+'.jpg'), target_size=(image_size, image_size))) 
                  for id in tqdm(labels['id'].values.tolist())])
print("Training shape: {}".format(train.shape))

#Sample dog images
fig = plt.figure(1, figsize=(8, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5)

for i in range(4):
    grid[i].imshow(train[i].astype(np.uint8))
    grid[i].set_title(le.classes_[train_labels[i]])

#split the training data into training and validation set
x_train, x_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.2, 
                                                  random_state=seed, stratify= train_labels)
print("Training features shape: {}".format(x_train.shape))
print("Validation features shape: {}".format(x_val.shape))

#one hot encoding of labels
NUM_OF_CLASSES = 120
y_train = to_categorical(y_train, num_classes=NUM_OF_CLASSES)
y_val = to_categorical(y_val, num_classes=NUM_OF_CLASSES)
print(y_train.shape)
print(y_val.shape)

#Data Augmentation
# Create train generator.
datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=45,#small angle for rotation
    zoom_range = 0.25, #for scalling
    width_shift_range=0.2,#translation
    height_shift_range=0.2,#translation
    horizontal_flip= True, #90 degree rotation
    vertical_flip = True, #90 degree rotation
    fill_mode = 'nearest'
)

# example of train generator will train the model taking a batch size
#we don't build a giant alterated image set for memory saving
train_generator = datagen.flow(
    x_train, y_train, shuffle=True, batch_size=9)


#example of the transformation
for x_batch, y_batch in train_generator:
    plt.figure(figsize = (10,10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(255*x_batch[i],interpolation='nearest',aspect='auto') #multplied by 255 to de-normalized the data
    plt.show()
    break

xception_model = Xception(include_top=False, weights='imagenet', pooling='avg')

#Extract feetures for training and validation images and save it into a file
xception_features_file = 'xception_features.npy'
xception_validationfeatures_file = 'xception_validationfeatures.npy'

if exists(xception_features_file):
    xception_features = np.load(xception_features_file)
    xception_validationfeatures = np.load(xception_validationfeatures_file)
else:
    xception_features = xception_model.predict(xception_preprocessor(x_train))
    xception_validationfeatures = xception_model.predict(xception_preprocessor(x_val))
    np.save(xception_features_file, xception_features)
    np.save(xception_validationfeatures_file, xception_validationfeatures)
    
print('Generated features shape: {}'.format(xception_features.shape))
print('Generated validation featurtes shape: {}'.format(xception_validationfeatures.shape))

#create a top model
top_model = Sequential()

#after global average pooling last hideen layer of our base model gives 2048-D vector for each image
top_model.add(Dense(1024, input_dim=2048, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(120, activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
top_model.compile(optimizer=SGD(lr=1e-2, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

epochs =100
batch_size = 64
top_model.fit(xception_features, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, callbacks=[early_stopping], validation_data=(xception_validationfeatures, y_val))

###### VGG16 ###################################################################################

seed = 1986
image_size = 224
train_dir = 'train'
test_dir = 'test'

#encode labels
le = LabelEncoder()
train_labels = le.fit_transform(labels['breed'])
labels['class'] = train_labels
labels.head()

#load training data
train = np.array([img_to_array(load_img(join(train_dir, id+'.jpg'), target_size=(image_size, image_size))) 
                  for id in tqdm(labels['id'].values.tolist())])
print("Training shape: {}".format(train.shape))

#split the training data into training and validation set
x_train, x_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.2, 
                                                  random_state=seed, stratify= train_labels)
print("Training features shape: {}".format(x_train.shape))
print("Validation features shape: {}".format(x_val.shape))

#one hot encoding of labels
NUM_OF_CLASSES = 120
y_train = to_categorical(y_train, num_classes=NUM_OF_CLASSES)
y_val = to_categorical(y_val, num_classes=NUM_OF_CLASSES)
print(y_train.shape)
print(y_val.shape)

vgg16_model = VGG16(include_top=False, weights='imagenet', pooling='avg')

#Extract feetures for training and validation images and save it into a file
vgg16_features_file = 'vgg16_features.npy'
vgg16_validationfeatures_file = 'vgg16_validationfeatures.npy'

if exists(vgg16_features_file):
    vgg16_features = np.load(vgg16_features_file)
    vgg16_validationfeatures = np.load(vgg16_validationfeatures_file)
else:
    vgg16_features = vgg16_model.predict(vgg16_preprocessor(x_train))
    vgg16_validationfeatures = vgg16_model.predict(vgg16_preprocessor(x_val))
    np.save(vgg16_features_file, vgg16_features)
    np.save(vgg16_validationfeatures_file, vgg16_validationfeatures)
    
print('Generated features shape: {}'.format(vgg16_features.shape))
print('Generated validation featurtes shape: {}'.format(vgg16_validationfeatures.shape))

#create a top model
top_model = Sequential()

#after global average pooling last hideen layer of our base model gives 1536-D vector for each image
top_model.add(Dense(1024, input_dim=512, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(120, activation='softmax'))

top_model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

epochs =100
batch_size = 64
top_model.fit(vgg16_features, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True,  validation_data=(vgg16_validationfeatures, y_val))

#### VGG 19 #####################################################################################

vgg19_model = VGG19(include_top=False, weights='imagenet', pooling='avg')

#Extract feetures for training and validation images and save it into a file
vgg19_features_file = 'vgg19_features.npy'
vgg19_validationfeatures_file = 'vgg19_validationfeatures.npy'

if exists(vgg19_features_file):
    vgg19_features = np.load(vgg19_features_file)
    vgg19_validationfeatures = np.load(vgg19_validationfeatures_file)
else:
    vgg19_features = vgg19_model.predict(vgg16_preprocessor(x_train))
    vgg19_validationfeatures = vgg19_model.predict(vgg16_preprocessor(x_val))
    np.save(vgg19_features_file, vgg19_features)
    np.save(vgg19_validationfeatures_file, vgg19_validationfeatures)
    
print('Generated features shape: {}'.format(vgg19_features.shape))
print('Generated validation featurtes shape: {}'.format(vgg19_validationfeatures.shape))

#create a top model
top_model = Sequential()

#after global average pooling last hideen layer of our base model gives 512-D vector for each image
top_model.add(Dense(1024, input_dim=512, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(120, activation='softmax'))

top_model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

epochs =100
batch_size = 64
top_model.fit(vgg19_features, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True,  validation_data=(vgg19_validationfeatures, y_val))

### ResNet50 ########################################################################################

resnet50_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

#Extract feetures for training and validation images and save it into a file
resnet50_features_file = 'resnet50_features.npy'
resnet50_validationfeatures_file = 'resnet50_validationfeatures.npy'

if exists(resnet50_features_file):
    resnet50_features = np.load(resnet50_features_file)
    resnet50_validationfeatures = np.load(resnet50_validationfeatures_file)
else:
    resnet50_features = resnet50_model.predict(vgg16_preprocessor(x_train))
    resnet50_validationfeatures = resnet50_model.predict(vgg16_preprocessor(x_val))
    np.save(resnet50_features_file, resnet50_features)
    np.save(resnet50_validationfeatures_file, resnet50_validationfeatures)
    
print('Generated features shape: {}'.format(resnet50_features.shape))
print('Generated validation featurtes shape: {}'.format(resnet50_validationfeatures.shape))

#create a top model
top_model = Sequential()

#after global average pooling last hideen layer of our base model gives 1536-D vector for each image
top_model.add(Dense(1024, input_dim=512, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(120, activation='softmax'))

top_model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

epochs =100
batch_size = 64
top_model.fit(resnet50_features, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True,  validation_data=(resnet50_validationfeatures, y_val))


## Sample Output ######################################################################################

pil_image = load_img('test/8895089b432e020cc42e6262b5a1c2dd.jpg', target_size=(image_size, image_size))
image = img_to_array(pil_image)
image = np.expand_dims(image, axis=0)
image_features = xception_model.predict(xception_preprocessor(image))
prediction = top_model.predict(image_features)
plt.imshow(pil_image)
print('predicted breed: {}'.format(le.classes_[np.argmax(prediction, axis=1)[0]]))

test_images = os.listdir(test_dir)
test = np.array([img_to_array(load_img(join(test_dir, img), target_size=(image_size, image_size))) 
                  for img in tqdm(test_images)])
print("Test shape: {}".format(test.shape))
test_features = xception_model.predict(xception_preprocessor(test))
test_predictions = top_model.predict(test_features)
result = pd.DataFrame(test_predictions)
result.columns = labels['breed'].sort_values().unique()
result['id'] = [img.split('.')[0] for img in test_images]
cols = result.columns.tolist()
cols = cols[-1:] + cols[:-1]
result = result[cols]

poster = result
poster = poster.drop("id",1)
print(poster.sort_values(by=4, ascending=False, axis=1).loc[4])







