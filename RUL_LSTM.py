#https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
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

def generate_rul_labeling(df):
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - train_df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

def create_time_window_labels(df, w0, w1):
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0 )
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df

def normalize(df):
    df['cycle_norm'] = df['cycle']
    cols_normalize = df.columns.difference(['id','cycle','RUL','label1','label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    return df

def visualize():
    engine_id3 = test_df[test_df['id'] == 3]
    engine_id3_50cycleWindow = engine_id3[engine_id3['RUL'] <= engine_id3['RUL'].min() + 50]
    cols1 = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    engine_id3_50cycleWindow1 = engine_id3_50cycleWindow[cols1]
    cols2 = ['s11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    engine_id3_50cycleWindow2 = engine_id3_50cycleWindow[cols2]

    # Plotting sensor data of engine 3 prior to a failure point
    ax1 = engine_id3_50cycleWindow1.plot(subplots=True, sharex=True, figsize=(20,20))
    ax2 = engine_id3_50cycleWindow2.plot(subplots=True, sharex=True, figsize=(20,20))
    plt.show()

# Reshape features into form (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# Function for generating labels
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

# RUL value of a specific enginge: 
# occurance with the highest cycle number of the same id - actual cycle number
def generate_rul_labeling(df):
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    # Merge max values with the training set
    df = df.merge(rul, on=['id'], how='left')
    # Calculate RUL
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df

def create_time_window_labels(df, w0, w1):
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0 )
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df

# MinMax normalization (from 0 to 1)
def normalize(df):
    df['cycle_norm'] = df['cycle']
    cols_normalize = df.columns.difference(['id','cycle','RUL','label1','label2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns = df.columns)
    return df

def generate_sequence_array(df):
    # Creating generator for the sequences
    seq_gen = (list(gen_sequence(df[df['id']==id], sequence_length, sequence_cols)) 
            for id in df['id'].unique())    

    # Generating sequences and converting them to numpy array
    seq_list = list(seq_gen)
    seq_list = list(filter(None, seq_list))
    seq_array = np.concatenate(seq_list).astype(np.float32)

    return seq_array

def introduce_ground_truth(filepath, test_df):
    # read ground truth data
    truth_df = pd.read_csv(filepath, sep=" ", header=None)

    # generate column max for test data
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)
    # generate RUL for test data
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    return test_df

def plot_results(predicted, actual):
    plt.plot(predicted)
    plt.plot(actual)
    plt.title('Results')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()

def generate_rul_label_arrays(df):
    label_gen = [gen_labels(df[df['id']==id], sequence_length, ['RUL']) 
                for id in df['id'].unique()]
    return np.concatenate(label_gen).astype(np.float32)

# Data preprocessing
training_set_path = 'phm08_data_set/train.txt'
test_set_path = 'phm08_data_set/test.txt'
path_of_ground_truth = ''

# Read training data
train_df = reading_dataset(training_set_path)

# Read test data
test_df = reading_dataset(test_set_path)

# 1. Data labeling - generating new column: RUL 

train_df = generate_rul_labeling(train_df)

if(path_of_ground_truth):
    test_df = introduce_ground_truth(path_of_ground_truth, test_df)
else:
    test_df = generate_rul_labeling(test_df)

# Generate labels for time windows (w1, w0)
w1 = 30
w0 = 15

train_df = create_time_window_labels(train_df, w0, w1)
test_df = create_time_window_labels(test_df, w0, w1)

# Feature scaling (min-max normalization)
train_df = normalize(train_df)
test_df = normalize(test_df)

sequence_length = 50

# Picking the feature columns
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm' ]
sequence_cols.extend(sensor_cols)

test_array = generate_sequence_array(test_df)  

seq_array = generate_sequence_array(train_df)

# Generating labels (RUL)
label_array = generate_rul_label_arrays(train_df)
label_test_array = generate_rul_label_arrays(test_df)

model = build_nn_for_regression(seq_array)
print(model.summary())

# Fit the network
history = model.fit(
     seq_array, label_array, epochs=50, batch_size=200, validation_split=0.05, verbose=1,
     callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto' )])

scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Mean Squared Error: {}'.format(scores[1]))
print('Mean Absolute Error: {}'.format(scores[2]))

# In-sample
predicted = model.predict(seq_array,verbose=1, batch_size=200)
actual = label_array

plot_results(predicted[0:1000], actual[0:1000])
plot_results(predicted, actual)

# Out-of-sample
scores_test = model.evaluate(test_array, label_test_array, verbose=2)
print('Mean Squared Error: {}'.format(scores_test[1]))
print('Mean Absolute Error: {}'.format(scores_test[2]))

predicted_test = model.predict(test_array)
actual_test = label_test_array

plot_results(predicted_test[0:1000], actual_test[0:1000])
plot_results(predicted_test, actual_test)
