import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def loading_data(filename):
    # loading dataset
    dataframe = pandas.read_csv(filename, delim_whitespace=True, header=None)
    dataset = dataframe.values
    X = dataset[:,0:13]
    Y = dataset[:,13]
    return X, Y

def baseline_model(input_dim):
    def bm():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(6, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    return bm

def evaluate( X, Y):
    estimator = KerasRegressor(build_fn=baseline_model(len(X[0])), epochs=50, batch_size=5, verbose=0)
    estimators = []
    # Standardization of input variables
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', estimator))

    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv = kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#file_name = "housing.csv"
#X, Y = loading_data(file_name)
#evaluate(estimator, X, Y)