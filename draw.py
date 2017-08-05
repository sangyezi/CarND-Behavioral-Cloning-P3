import matplotlib.pyplot as plt

# from keras.models import load_model
# from keras.utils.visualize_util import model_to_dot

# model = load_model('model.h5')

# plot_model(model, to_file='model.png')
# from IPython.display import SVG
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

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