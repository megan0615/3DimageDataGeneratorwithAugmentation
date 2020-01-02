import numpy as np
import keras
from scipy import ndimage


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=5, dim=(61,73,61), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        orig_X, orig_y = self.__data_generation(list_IDs_temp)
        aug_X, aug_y =self.__data_augmentation(list_IDs_temp)
        X = np.concatenate((orig_X,aug_X),aixs=0)
        y = np.concatenate((orig_y,aug_y),aixs=0)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    def random_rotate(x):
        angle=random.randint(-20,20)
        axes= random.sample((0,1,2),k=2)
        rotated_x=ndimage.rotate(x, angle=angle, axes=axes, reshape=False, order=3, mode='constant')
        return rotated_x
    def random_shift(x):
        shift=[random.randint(-10,10),random.randint(-10,10),random.randint(-10,10)]
        shifted_x=ndimage.shift(x, shift=shift, order=3, mode='constant', cval=0.0)
        return shifted_x
    def __data_augmentation(self,list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        shifted_X = np.empty((self.batch_size, self.n_channels, *self.dim))
        rotated_X = np.empty((self.batch_size, self.n_channels, *self.dim))
        shiftrotated_X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')
            shifted_X[i] = random_shift(X[i])
            roated_X[i] = random_rotate(X[i])
            shiftrotated_X = random_roate(shifted_X[i])            

            # Store class
            y[i] = self.labels[ID]
        aug_X=np.concatenate((shifted_X, rotated_X, shiftrotated_X),axis=0)
        aug_y=np.concatenate((y,y,y),axis=0)
        return aug_X, aug_y
        
