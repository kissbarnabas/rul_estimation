# Reference: https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
import sys
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Setting seed for reproducability
np.random.seed(1234)  
PYTHONHASHSEED = 0
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation

def reading_dataset(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None)
    df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
    df = df.sort_values(['id','cycle'])
    return df

# Generates the RUL label column for the dataset
def generate_rul_labeling(df):
    # RUL value of a specific enginge: 
    # occurance with the highest cycle number of the same id - actual cycle number
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']

    # Merge max values with the training set
    df = df.merge(rul, on=['id'], how='left')

    # Calculate RUL
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

# Performs MinMax normalization to range 0-1
def normalize(df):
    # Defining a new column for feature normalized cycle number
    df['cycle_norm'] = df['cycle']

    # Columns to be normalized
    cols_normalize = df.columns.difference(['id','cycle','RUL'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    return df

# Reshapes features into form (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# Generates array containing the labels of the specified chunk
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# Building the network
# An LSTM layer with 100 units followed by another with 50 units
# Dropout of 0.2 applied after each LSTM layer
# Regression -> final layer: a single unit with linear activation
# Stochastic gradient descent optimizer used
def build_nn_for_regression(seq_array):
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]
    
    model = Sequential()

    model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            units=50,
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=nb_out, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse','mae'])
    print(model.summary())
    return model

# Generates sequences of specified length with specified columns
def generate_sequence_array(df, sequence_length, sequence_cols):
    # Creating generator for the sequences
    seq_gen = (list(gen_sequence(df[df['id']==id], sequence_length, sequence_cols)) 
            for id in df['id'].unique())    

    # Generating sequences and converting them to numpy array
    seq_list = list(seq_gen)
    seq_list = list(filter(None, seq_list))
    seq_array = np.concatenate(seq_list).astype(np.float32)

    return seq_array

def plot_results(predicted, actual):
    plt.plot(predicted)
    plt.plot(actual)
    plt.title('Results')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()

# Generate arrays containing the RUL labels for learning
def generate_rul_label_arrays(df):
    label_gen = [gen_labels(df[df['id']==id], sequence_length, ['RUL']) 
                for id in df['id'].unique()]
    return np.concatenate(label_gen).astype(np.float32)

# Data preprocessing
training_set_path = 'phm08_data_set/train.txt'
test_set_path = 'phm08_data_set/test.txt'

# Reading training data
train_df = reading_dataset(training_set_path)

# Reading test data
test_df = reading_dataset(test_set_path)

# 1. Data labeling - generating new column: RUL 
train_df = generate_rul_labeling(train_df)
test_df = generate_rul_labeling(test_df)

# Feature scaling (min-max normalization)
train_df = normalize(train_df)
test_df = normalize(test_df)

# Picking the feature columns
# The features used as inputs are the 21 sensor measurements
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm' ]
sequence_cols.extend(sensor_cols)

sequence_length = 50

# Generating the sequences 
test_array = generate_sequence_array(test_df, sequence_length, sequence_cols)  
seq_array = generate_sequence_array(train_df, sequence_length, sequence_cols)

# Generating labels (RUL)
label_array = generate_rul_label_arrays(train_df)
label_test_array = generate_rul_label_arrays(test_df)

# Building the neural network
model = build_nn_for_regression(seq_array)
print(model.summary())

# Fit the network
history = model.fit(
     seq_array, label_array, epochs=50, batch_size=200, validation_split=0.05, verbose=1,
     callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto' )])

# Evaluating the model's accuracy on the traning data
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Mean Squared Error: {}'.format(scores[1]))
print('Mean Absolute Error: {}'.format(scores[2]))

# In-sample test - prediction on the training data
predicted = model.predict(seq_array,verbose=1, batch_size=200)
actual = label_array

# Plotting results
plot_results(predicted[0:1000], actual[0:1000])
plot_results(predicted, actual)

# Evaluating the model's accuracy on the test data
scores_test = model.evaluate(test_array, label_test_array, verbose=2)
print('Mean Squared Error: {}'.format(scores_test[1]))
print('Mean Absolute Error: {}'.format(scores_test[2]))

# Out-of-sample test - prediction on the test data
predicted_test = model.predict(test_array)
actual_test = label_test_array

# Plotting results
plot_results(predicted_test[0:1000], actual_test[0:1000])
plot_results(predicted_test, actual_test)
