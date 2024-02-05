# Download the datasets from
# ** https://github.com/cerndb/SparkDLTrigger/tree/master/Data **
#
# For CERN users, data is already available on EOS
PATH = "C:/Users/adoko/data/"


def convert_vector_to_list(vector):
    return list(vector)
if __name__ == '__main__':

    import pandas as pd

    testPDF = pd.read_parquet(path= PATH + 'testUndersampled.snappy.parquet',
                              columns=['HLF_input', 'encoded_label'])

    trainPDF = pd.read_parquet(path= PATH + 'trainUndersampled.snappy.parquet',
                               columns=['HLF_input', 'encoded_label'])


    # Define a function to convert vectors to arrays (lists in this case)
     # Adjust this based on the actual structure of your vectors


    # Transforming df_test
    df_test = testPDF.copy()
    df_test['HLF_input'] = df_test['HLF_input'].apply(convert_vector_to_list)
    df_test['encoded_label'] = df_test['encoded_label'].apply(convert_vector_to_list)
    testPDF = df_test[['HLF_input', 'encoded_label']]

    # Print the data types (schema) in Pandas
    print(df_test.dtypes)

    # Transforming df_train
    df_train = trainPDF.copy()
    df_train['HLF_input'] = df_train['HLF_input'].apply(convert_vector_to_list)
    df_train['encoded_label'] = df_train['encoded_label'].apply(convert_vector_to_list)
    trainPDF = df_train[['HLF_input', 'encoded_label']]



    # Check the number of events in the train and test datasets

    num_test = testPDF.count()
    num_train = trainPDF.count()

    print('There are {} events in the test dataset'.format(num_test))
    print('There are {} events in the train dataset'.format(num_train))


    import numpy as np

    X = np.stack(trainPDF["HLF_input"])
    y = np.stack(trainPDF["encoded_label"])

    X_test = np.stack(testPDF["HLF_input"])
    y_test = np.stack(testPDF["encoded_label"])

    import tensorflow as tf
    tf.__version__

    # Check that we have a GPU available
    tf.config.list_physical_devices('GPU')


    from keras.optimizers import Adam
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    def create_model(nh_1, nh_2, nh_3):
        ## Create model
        model = Sequential()
        model.add(Dense(nh_1, input_shape=(14,), activation='relu'))
        model.add(Dense(nh_2, activation='relu'))
        model.add(Dense(nh_3, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        ## Compile model
        optimizer = 'Adam'
        loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

        return model

    keras_model = create_model(50,20,10)

    batch_size = 128
    n_epochs = 5

    history = keras_model.fit(X, y, batch_size=batch_size, epochs=n_epochs, \
                                    validation_data=(X_test, y_test))


    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    # Graph with loss vs. epoch

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title("HLF classifier loss")
    plt.show()