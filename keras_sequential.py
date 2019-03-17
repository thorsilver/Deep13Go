import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# Load sample data into Numpy arrays
np.random.seed(123)
X = np.load('generated_games/features-40k.npy')
Y = np.load('generated_games/labels-40k.npy')
samples = X.shape[0]
board_size = 9 * 9
input_shape = (board_size, board_size, 1)

# Transform input into length-81 vector
X = X.reshape(samples, board_size)
Y = Y.reshape(samples, board_size)

# 90% training set, 10% test set
train_samples = 10000
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

# define a multilayer perceptron
model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(board_size,)))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(board_size, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=100,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
