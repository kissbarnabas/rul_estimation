#https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
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

# Read training data
train_df = pd.read_csv('PM_train.txt', sep=' ', header=None)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])

# Read test data
test_df = pd.read_csv('PM_test.txt', sep=' ', header=None)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
                    
# Read ground truth data
truth_df = pd.read_csv('PM_truth.txt', sep=' ', header=None)


# Data preprocessing

# 1. Data labeling - generating new column: RUL 
# RUL value of a specific enginge: 
    # occurance with the highest cycle number of the same id - actual cycle number
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

# Merge max values with the training set
train_df = train_df.merge(rul, on=['id'], how='left')
# Calculate RUL
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# Generate labels for time windows (w1, w0)
w1 = 30
w0 = 15

train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

# Feature scaling (min-max normalization)


# MinMax normalization
train_df['cycle_norm'] = train_df['cycle']

# Columns to be normalized: all columns except for id, cycle, RUL, label1, label2
cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)
test_df.head()

# Labeling test data
# Create column max for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index+1
truth_df['max'] = rul['max'] + truth_df['more'] #???????
truth_df.drop('more', axis=1, inplace=True)

# Generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# Generate labels for time windows for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

sequence_length = 50

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
  
#visualize()

# Reshape features into form (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# Picking the feature columns
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm' ]
sequence_cols.extend(sensor_cols)

# Creating generator for the sequences
seq_gen = (list(
    gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)
    ) for id in train_df['id'].unique() ) 

# Generating sequences and converting to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# Function for generating labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# Generating labels
# changed: from label1 to RUL
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
print("Label array: ", label_array)
# Building the network
# An LSTM layer with 100 units followed by another with 50 units
# Dropout of 0.2 applied after each LSTM layer
# Binary classification -> final layer: a single unit with sigmoid activation
def build(seq_array):
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

    model.add(Dense(units=nb_out, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

model = build(seq_array)

# Fit the network
model.fit(
     seq_array, label_array, epochs=10, batch_size=200, validation_split=0.05, verbose=1,
     callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto' )])

scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))

#y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)
y_true = label_array
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
#cm = confusion_matrix(y_true, y_pred)
#print(cm)

#precision = precision_score(y_true, y_pred)
#recall = recall_score(y_true, y_pred)
#print( 'precision = ', precision, '\n', 'recall = ', recall)

# Performance on the test data
# For testing: only the last sequence for each id will be kept
seq_array_test_last = [
    test_df[test_df['id']==id][sequence_cols].values[-sequence_length:]
            for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]

