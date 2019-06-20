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
idx = np.random.choice(X_train.shape[0],8,replace=False)
for i,ix in enumerate(idx):
    plt.subplot(241+i)
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