import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks')
# Load Dataset from Keras Libraries
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_class = 10

# Visualize Some Examples
idx = np.random.choice(X_train.shape[0],6,replace=False)
for i,ix in enumerate(idx):
    plt.subplot(231+i)
    plt.title('Label is {0}'.format(y_train[ix]))
    plt.imshow(X_train[ix], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Reshape X data from 2D to 1D (28x28 to 784)
X_train = X_train.reshape(60000,784).astype('float32')
X_test  = X_test.reshape(10000,784).astype('float32')

# Convert Y labels to Categorical One-Hot Format
y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)
#########
#########
#########
# Import Keras Libraries
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Define Model Parameters
nb_feat   = 28    # no. of features/columns of input
L1_units  = 256   # no. of nodes in Layer 1
L2_units  = 100   # no. of nodes in Layer 2
L3_units  = 50   # no. of nodes in Layer 2
nb_class  = 10    # no. of output classes

# Neural Network Model
model = Sequential()                             # Sequential network model description
model.add(Dense(L1_units,input_shape=(784,)))    # Add 1st Dense Layer
model.add(Activation("relu"))                    # Add activation function

model.add(Dense(L2_units))                       # Add 2nd Dense Layer
model.add(Activation("relu"))                    # Add activation function

model.add(Dense(L3_units))                       # Add 3nd Dense Layer
model.add(Activation("relu"))                    # Add activation function

model.add(Dense(nb_class))                       # Add 3rd Dense Layer, also the classification layer
model.add(Activation('softmax'))                 # Add sigmoid classification
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
log = model.fit(X_train, y_train,nb_epoch=15, batch_size=128, verbose=2,validation_data=(X_test, y_test))
###########
# Scikit-Learn Machine Learning Utilities
from sklearn.metrics import confusion_matrix

## Final Accuracy
score = model.evaluate(X_test, y_test)
print('Model Accuracy: {}%'.format(score[1]*100))
print ('')
print ('')

y_pred = model.predict_classes(X_test)
cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)  ## Confusion matrix
plot_confusion_matrix(cm, ['0','1','2','3','4','5','6','7','8','9'],normalize=True)