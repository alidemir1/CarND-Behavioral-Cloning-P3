import csv
from scipy import ndimage
import numpy as np

samples = []
with open('../../../opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

        
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from sklearn.utils import shuffle  
    

def data_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name_appendix = batch_sample[i].split('/')[-1]
                    if name_appendix == 'center':
                        break
                    name = '../../../opt/carnd_p3/data/IMG/' + name_appendix
                    center_image = ndimage.imread(name)
                    center_angle = float(batch_sample[3])
                    correction = 0.2
                    images.append(center_image)
                    if i == 1:
                        center_angle += correction
                    elif i == 2:
                        center_angle -= correction
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
    
batch_size = 32

# compile and train the model using the generator function
train_generator = data_generator(train_samples, batch_size=batch_size)
validation_generator = data_generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
model.add(Activation('relu',name='relu2'))
model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

model.save('model.h5')

exit()