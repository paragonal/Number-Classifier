from mnist_classifier.data_reader import *
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import load_model
import numpy as np
import time
from keras.layers import Flatten

training_inputs, training_outputs = None, None
test_inputs, test_outputs = None, None
images,outputs = None, None

def create_model():
    optimizer = keras.optimizers.Adam(learning_rate=.0025)
    m = Sequential()
    m.add(Conv2D(32, kernel_size=(4,4),activation='relu', input_shape=(28,28,1)))
    m.add(MaxPooling2D(pool_size=(3,3)))
    m.add(Dropout(.1))
    m.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m.add(GlobalAveragePooling2D(data_format='channels_last'))
    m.add(Dropout(.1))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer = optimizer)
    return m

def load_data():
    labels = get_labels("../train-labels-idx1-ubyte")
    outputs = []
    for label in labels:
        row = [0 for i in range(10)]
        row[label] = 1
        outputs.append(row)
    outputs = np.array(outputs)

    images = np.array(get_images("../train-images-idx3-ubyte"))
    images = images.reshape(60000,28,28,1)
    training_outputs = outputs[:55000]
    training_inputs = images[:55000]

    test_outputs = outputs[55000:]
    test_inputs = images[55000:]



#model = create_model()
#print(model.summary())

def fit_model(model):
    history = model.fit(
        training_inputs,
        training_outputs,
        batch_size=100,
        epochs=10,
        validation_data=(test_inputs, test_outputs),
    )
    model.save("classification_model")
    return model



def eval_model(model):
    failed = 0
    guesses = [np.argmax(x) for x in model.predict(images)]
    for i in range(60000):
        if guesses[i] != np.argmax(outputs[i]):
            failed+=1
    print("Error Rate:", failed/60000)
    return failed/60000

def blend(list_images):

    equal_fraction = 1.0 / (len(list_images))

    output = np.zeros_like(list_images[0])

    for img in list_images:
        output = output + img * equal_fraction

    output = output.astype(np.uint8)
    return output

vid = cv2.VideoCapture(0)
model = load_model("classification_model")

while (True):

    #time.sleep()
    frames = []
    for i in range(10):
        ret, frame = vid.read()
        frames.append(frame)

    frame = blend(frames)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, frame) = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)
    frame=255-frame
    nninput = cv2.resize(frame, (28,28)).reshape(1,28,28,1)
    guess = [np.argmax(x) for x in model.predict(nninput)][0]
    print(guess)

    imout = cv2.resize(frame, (256,256))

    cv2.imshow("frame", imout)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("images/camera_image.png", nninput)
vid.release()
cv2.destroyAllWindows()