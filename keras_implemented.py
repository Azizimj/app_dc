import tensorflow
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(110)
IMG_HEIGHT = 64  # the image height to be resized to
IMG_WIDTH = 64  # the image width to be resized to
CHANNELS = 1

training_prec = 0.8 # trian set percentage
eval_prec = .1  # validation set percentage

def augmentation(image):
    result = []
    angels = np.random.random(5)*50-25  # random -25, 25 degree rotate
    for angel in angels:
        result.append(cv2.warpAffine(image, cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_HEIGHT / 2), angel, 1.0),dsize=(IMG_WIDTH,IMG_HEIGHT)))
    result.append(cv2.flip(image, flipCode=-1)) # vertical and horizontal flip
    result.append(cv2.flip(image, flipCode=0)) # vertical flip
    result.append(cv2.flip(image, flipCode=1)) # horizontal flip
    invGamma = 1
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    result.append(cv2.LUT(image, table))

    return result

def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0

    for labels in os.listdir(directory):  # Main Directory where each class label is present as folder name.
        if labels == 'defected':  # Folder contain Glacier Images get the '2' class label.
            label = 1
        elif labels == 'undefected':
            label = 0

        for image_file in os.listdir(directory + labels):  # Extracting the file name of the image from Class Label folder
            # image = cv2.imread(directory + labels + r'/' + image_file)  # Reading the image (OpenCV)
            image = cv2.imread(directory + labels + r'/' + image_file, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            Images.append(image)
            Labels.append(label)
            if label == 1:
                res = augmentation(image)
                Images += res
                Labels += [label]*len(res)


    Images, Labels = shuffle(Images, Labels, random_state=817328462)
    Images = np.array(Images)
    Labels = np.array(Labels)
    num_data = len(Labels)

    num_training = int(num_data * training_prec)
    num_eval = int(num_data * eval_prec)

    tr_X = Images[:num_training]
    eval_X = Images[num_training: num_eval + num_training]
    tes_X = Images[num_eval + num_training:]

    tr_y = Labels[:num_training]
    eval_y = Labels[num_training: num_training + num_eval]
    tes_y = Labels[num_training + num_eval:]

    return tr_X, eval_X, tes_X, tr_y, eval_y, tes_y


def get_classlabel(class_code):
    labels = {1: 'defected', 0: 'undefected'}

    return labels[class_code]


Images, eval_images, test_images,\
Labels, eval_labels, test_labels = get_images('splitted-all/') #Extract the training images from the folders.

defection_prec = lambda x: (len(x), sum(x)/len(x))
print('Train set has %d sample and defection proportion = %.3f' % defection_prec(Labels))
print('Validation set has %d sample and defection proportion = %.3f' % defection_prec(eval_labels))
print('Test set has %d sample and defection proportion = %.3f' % defection_prec(test_labels))

if CHANNELS == 1:
    Images = np.expand_dims(Images, axis=-1)
    eval_images = np.expand_dims(eval_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)


model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',
                        input_shape=(IMG_HEIGHT,IMG_WIDTH,CHANNELS)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
# model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
# model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
# model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax'))


l_r = tensorflow.train.exponential_decay(learning_rate=1e-3, global_step=0,
                                         decay_steps=50, decay_rate=0.9, staircase=True)
optimer = tensorflow.train.AdamOptimizer(l_r)

# model.compile(optimizer=Optimizer.Adam(lr=1e-4),
#               loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=optimer,
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.summary()


aug = ImageDataGenerator(rotation_range=2, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, brightness_range=(-1,2))

# zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest",

# trained = model.fit(aug.flow(Images, Labels, batch_size=10), epochs=5,
#                     validation_data=(eval_images, eval_labels))
trained = model.fit(Images, Labels, epochs=5,
                    validation_data=(eval_images, eval_labels))


plot.plot(trained.history['acc'])
plot.plot(trained.history['val_acc'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()


test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("model evaluated on test")
model.evaluate(test_images, test_labels, verbose=1)
