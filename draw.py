import matplotlib.pyplot as plt
import pickle

from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('model.h5')
plot(model, to_file='model.png')

history = pickle.load(open('history_object.p', 'rb'))

### print the keys contained in the history object
print(history.keys())

### plot the training and validation loss for each epoch
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()