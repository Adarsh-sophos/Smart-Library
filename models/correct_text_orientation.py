import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def get_data(type_of_data):
    images = []

    if(type_of_data == "train"):
        files = os.listdir("IIIT5K/train/")

    else:
        files = os.listdir("IIIT5K/test/")

    for image_name in files:

        if(type_of_data == "train"):
            img_path = os.path.join("IIIT5K/train", image_name)

        else:
            img_path = os.path.join("IIIT5K/test", image_name)
        
        image = cv2.imread(img_path)
        
        r = 32.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 32)
        
        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        #plt.imshow(resized)
        
        #print(image.shape)
        #print(resized.shape)
        r, c, _ = resized.shape
        
        if(c >= 64):
            image = resized[:, :64]
            #plt.imshow(image)
            
            images.append(image)
            
        
    X = np.array(images)
    print(X.shape)

    return X


def get_labels(X):
    Y = []

    for i in range(X.shape[0]):
        Y.append(1)

    Y = np.array(Y).reshape(X.shape[0], 1)

    return Y


def randomly_rotate_images(X, Y):
    index = np.random.choice(X.shape[0], int(X.shape[0] / 2), replace=False)
    #print(len(index))

    for i in index:
        X[i] = imutils.rotate(X[i], -180)
        Y[i] = 0

    return X, Y


def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model


def evaluate_model(X, Y):
    preds = model.evaluate(x = X, y = Y)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


all_images = get_data("train")
all_images.concatenate(get_data("test"))
print(all_images.shape)

Y = get_labels(all_images)

all_images, Y = randomly_rotate_images(all_images, Y)

X_train, X_test, y_train, y_test = train_test_split(all_images, Y, test_size=0.33, shuffle = True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = model((32, 64, 3))

model.compile(loss='binary_crossentropy', optimizer = 'Adam' , metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 3, batch_size = 64)

evaluate_model(X_train, y_train)
evaluate_model(X_test, y_test)

model.save('text_orientation_model.h5')

history_dict = history.history
history_dict.keys()

loss_values = history_dict['loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo')
plt.xlabel('Epochs')
plt.ylabel('Loss')