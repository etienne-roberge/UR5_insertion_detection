from keras import layers
from keras import models
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import DatasetManager_


#Model definition
def conv_network(x):

    x = layers.Conv1D(16, kernel_size=(3), strides=(1), padding='same')(x)
    x = layers.BatchNormalization(axis=-1,momentum=0.3)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(32, kernel_size=(3), strides=(1), padding='same')(x)
    x = layers.BatchNormalization(axis=-1,momentum=0.3)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(64, kernel_size=(3), strides=(1), padding='same')(x)
    x = layers.BatchNormalization(axis=-1,momentum=0.3)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(128, kernel_size=(3), strides=(1), padding='same')(x)
    x = layers.BatchNormalization(axis=-1,momentum=0.3)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(500)(x)
    x = layers.BatchNormalization(axis=-1, momentum=0.3)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dense(units=3, kernel_initializer="he_normal",activation="softmax")(x)

    return x


batch_size = 100
nb_epoch = 200
img_height = 1
img_width = 1000
img_channels = 9

#Define the input tensor size
image_tensor = layers.Input(shape=(img_width, img_channels))

#Define the model
network_outputx = conv_network(image_tensor)
model = models.Model(inputs=[image_tensor], outputs=[network_outputx])
print(model.summary())

#Using the Dataset helper class to create folds and test set
dataset = DatasetManager_.DataSet(newDataBaseName="Insertion_database_16March2018.dataset",
                                  saveDatabase=True, saveName="ready-to-use-dataset.dat")
#You can also load a DatasetManager class dataset with:
#dataset = DatasetManager_.DataSet(loadDatabase_savepoint=True, loadName="ready-to-use-dataset.dat")


#init dataset training and validation
dataset.init_fold_data(fold_num=0)

#Labels for training/validation
Y_train = dataset._train_label_KFold
Y_valid = dataset._valid_label_KFold

#Signals for training/validation
X_train = dataset._train_data_KFold
X_train = np.reshape(X_train,(-1, img_width, img_channels))
X_valid = dataset._valid_data_KFold
X_valid = np.reshape(X_valid,(-1, img_width, img_channels))

#Optimiser parameters
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=6)
csv_logger = CSVLogger('training_info.csv')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Train
model.fit(x = X_train,
             y = Y_train,
             batch_size=batch_size,
             epochs=nb_epoch,
             validation_data=(X_valid,Y_valid),
             shuffle=True,
             callbacks=[lr_reducer, early_stopper, csv_logger])

model.save('model.h5')
