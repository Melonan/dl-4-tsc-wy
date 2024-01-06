# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
from  tensorflow import keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True,lr=0.001):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes,lr=lr)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return


    def build_model(self, input_shape, nb_classes,lr):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:  # for italypowerondemand dataset
            padding = 'same'
        l2_reg = keras.regularizers.l2(1e-4)
        # First convolutional layer with Batch Normalization
        conv1 = keras.layers.Conv1D(filters=16, kernel_size=7, padding=padding, kernel_regularizer=l2_reg)(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        conv1 = keras.layers.Dropout(0.5)(conv1)

        # Second convolutional layer
        conv2 = keras.layers.Conv1D(filters=32, kernel_size=7, padding=padding,kernel_regularizer=l2_reg)(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        conv2 = keras.layers.Dropout(0.5)(conv2)
        
        flatten_layer = keras.layers.Flatten()(conv2)

        # Dense layer with Batch Normalization
        dense = keras.layers.Dense(units=64)(flatten_layer)
        dense = keras.layers.BatchNormalization()(dense)
        dense = keras.layers.ReLU()(dense)
        dense = keras.layers.Dropout(0.5)(dense)

        # Output layer
        activation = 'softmax' if nb_classes > 2 else 'sigmoid'
        output_layer = keras.layers.Dense(units=nb_classes, activation=activation)(dense)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # # Adding L2 regularization
        # l2_reg = keras.regularizers.l2(1e-4)

        # # Using ReLU activation and adding Dropout
        # conv1 = keras.layers.Conv1D(filters=6, kernel_size=7, padding=padding, 
        #                             activation='relu', kernel_regularizer=l2_reg)(input_layer)
        # conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)
        # conv1 = keras.layers.Dropout(0.5)(conv1)  # Adding dropout

        # conv2 = keras.layers.Conv1D(filters=12, kernel_size=7, padding=padding, 
        #                             activation='relu', kernel_regularizer=l2_reg)(conv1)
        # conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)
        # conv2 = keras.layers.Dropout(0.5)(conv2)  # Adding dropout

        # flatten_layer = keras.layers.Flatten()(conv2)

        # output_layer = keras.layers.Dense(units=nb_classes, activation='sigmoid', 
        #                                 kernel_regularizer=l2_reg)(flatten_layer)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        
        # Compile the model
        loss = 'categorical_crossentropy' if nb_classes > 2 else 'binary_crossentropy'
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
        #             metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
                                                    min_lr=0.000015)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                        save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, 
                                                    restore_best_weights=True)

        self.callbacks = [ model_checkpoint,reduce_lr]
        return model

    # def build_model(self, input_shape, nb_classes):
    #     padding = 'valid'
    #     input_layer = keras.layers.Input(input_shape)

    #     if input_shape[0] < 60: # for italypowerondemand dataset
    #         padding = 'same'

    #     conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
    #     conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

    #     conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
    #     conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

    #     flatten_layer = keras.layers.Flatten()(conv2)

    #     output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

    #     model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    #     model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
    #                   metrics=['accuracy'])

    #     file_path = self.output_directory + 'best_model.hdf5'
    #     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
	# 	min_lr=0.0001)
    #     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
    #                                                        save_best_only=True)
    #     early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

    #     self.callbacks = [reduce_lr,model_checkpoint,early_stopping]
    #     return model

    def fit(self, x_train, y_train, x_val, y_val, y_true,batch_size=16, epochs=500):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        
        
        nb_epochs = epochs

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()
        return hist

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
