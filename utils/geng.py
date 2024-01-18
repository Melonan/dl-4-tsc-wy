import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils.utils import create_directory
import logging

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose,lr=0.0005)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


def fit_splits(classifier, X, Y, batch_size=16, epochs=500, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    histories = []
    ori_output_directory = classifier.output_directory

    for train_index, val_index in kf.split(X):
        x_train_fold, x_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]
        mini_batch_size = int(min(x_train_fold.shape[0]/10, batch_size))
        logging.info(f"fold {fold_no}: x_train_fold.shape={x_train_fold.shape}, y_train_fold.shape={y_train_fold.shape}")

        output_directory_this_fold = ori_output_directory + 'fold_'+str(fold_no)+'/'
        classifier.output_directory = output_directory_this_fold

        create_directory(output_directory_this_fold)
        classifier.model = classifier.build_model(X.shape[1:], Y.shape[1],lr=0.00025)
        classifier.model.summary()
        classifier.model.load_weights(ori_output_directory + 'model_init.hdf5')

        y_true_fold = np.argmax(y_val_fold, axis=1)

        history = classifier.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, y_true_fold, batch_size=mini_batch_size, epochs=epochs)
        histories.append((fold_no, history))

        fold_no += 1

    summarize_results(histories)  # This function remains the same


def fit_splits_for_mcnn(classifier, X, Y, epochs=300, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    histories = []
    ori_output_directory = classifier.output_directory

    for train_index, val_index in kf.split(X):
        x_train_fold, x_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]
        # mini_batch_size = int(min(x_train_fold.shape[0]/10, batch_size))
        logging.info(f"fold {fold_no}: x_train_fold.shape={x_train_fold.shape}, y_train_fold.shape={y_train_fold.shape}")

        output_directory_this_fold = ori_output_directory + 'fold_'+str(fold_no)+'/'
        classifier.output_directory = output_directory_this_fold

        create_directory(output_directory_this_fold)

        # classifier.model = classifier.build_model(X.shape[1:], Y.shape[1],lr=0.00025)
        # classifier.model.summary()
        # classifier.model.load_weights(ori_output_directory + 'model_init.hdf5')

        y_true_fold = np.argmax(y_val_fold, axis=1)

        history = classifier.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, y_true_fold, epochs=epochs)
        histories.append((fold_no, history))

        fold_no += 1

    summarize_results(histories)  # This function remains the same





def summarize_results(histories):
    total_loss = []
    total_acc = []
    total_val_loss = []
    total_val_acc = []

    # 遍历所有历史记录
    for _,history in histories:
        # 计算每一折的平均损失和准确率
        total_loss.append(np.mean(history.history['loss']))
        total_acc.append(np.mean(history.history['accuracy']))
        total_val_loss.append(np.mean(history.history['val_loss']))
        total_val_acc.append(np.mean(history.history['val_accuracy']))

    # 打印结果
    logging.info(f'Average training loss: {np.mean(total_loss)}')
    logging.info(f'Average training accuracy: {np.mean(total_acc)}')
    logging.info(f'Average validation loss: {np.mean(total_val_loss)}')
    logging.info(f'Average validation accuracy: {np.mean(total_val_acc)}')
    

def standard_scaler_total(X):
    # Assuming X is a numpy array with shape (360, 45, 2)
    # (n_cases, n_timepoints, n_channels)
    # For demonstration, let's create a sample array with this shape
    # X = np.random.rand(360, 45, 2)

    # Initialize a StandardScaler object
    scaler = StandardScaler()

    # Reshape the data for StandardScaler
    X_reshaped = X.reshape(-1, X.shape[-1])  # Reshaping to (360*45, 2)

    # Fit the scaler on the data and transform
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to original shape (360, 45, 2)
    X_scaled = X_scaled.reshape(X.shape)

    X_scaled.shape  # Verifying the shape after scaling
    return X_scaled

def standard_scaler_individual(X_sample):
    # Sample data: 3D numpy array with shape (360, 45, 2)
    # X_sample = np.random.rand(360, 45, 2)

    # Initialize an empty array to store the scaled data
    X_scaled_individual = np.zeros_like(X_sample)

    # Iterate over each case/sample in the dataset
    for i in range(X_sample.shape[0]):
        # Reshape the data for StandardScaler (flattening timepoints and channels)
        # 这一句有和没有没区别
        sample_reshaped = X_sample[i].reshape(-1, X_sample.shape[-1])
        # logging.info(f'sample_reshaped.shape:{sample_reshaped.shape}')
        # Initialize a new scaler for each sample
        scaler_individual = StandardScaler()
        # Fit the scaler on the data and transform
        sample_scaled = scaler_individual.fit_transform(sample_reshaped)

        # Reshape back to the original shape and store in the scaled array
        X_scaled_individual[i] = sample_scaled.reshape(X_sample[i].shape)
        break
        

    X_scaled_individual.shape  # Verifying the shape after scaling
