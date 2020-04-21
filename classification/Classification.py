import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def load_data_example():
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

    # input
    X = dataset[:, 0:8]
    # output
    y = dataset[:, 8]
    return X, y



#define keras model
def classify(X, y):
    model = Sequential()
    model.add(Dense(12, input_dim=len(X[0]), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=150, batch_size=10)

    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    predictions = model.predict_classes(X)

    sum = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            sum = sum + 1

    print(sum/len(predictions))
