import numpy as np
import keras
from scipy import ndimage

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=5, dim=(61,73,61), n_channels=1,
                 n_classes=2, shuffle=True):
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

        # load original data
        orig_X, orig_Y = self.__data_generation(list_IDs_temp)
        # perform augmentation
        aug_X, aug_Y =self.__data_augmentation(list_IDs_temp)
        
        #combine original and sumented data
        X = np.concatenate((orig_X,aug_X),aixs=0)
        Y = np.concatenate((orig_Y,aug_Y),aixs=0)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]
        Y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, Y
    
    def random_rotate(x):
        'random rotate image along random axis within range [-20, 20]'
        #define random angle
        angle = random.randint(-20,20)
        #define random axis
        axes = random.sample((0,1,2),k=2)
        
        #perform rotation
        rotated_x = ndimage.rotate(x, angle=angle, axes=axes, reshape=False, order=3, mode='constant')      
        return rotated_x
    
    def random_shift(x):
        'random shift image along three axes within range [-10, 10]'
        #define random shift value
        shift = [random.randint(-10,10),random.randint(-10,10),random.randint(-10,10)]
        
        #perform shift
        shifted_x = ndimage.shift(x, shift=shift, order=3, mode='constant', cval=0.0)        
        return shifted_x

    
    def random_zoom_and_crop(x):
        'random zoom image along three axes within range [0.9, 1.1]'
        #define zoom value
        zoom = [random.uniform(0.9,1.1),random.uniform(0.9,1.1),random.uniform(0.9,1.1)]
        
        #perform zoom
        zoomed_x = ndimage.zoom(x, zoom, order=3, mode='constant', cval=0.0)
        
        #padding and crop
        orig_shape = x.shape
        zoom_shape = zoomed_x.shape
        
        if zoom[0] < 1:
            padding_width=orig_shape[0]-zoom_shape[0]
            z0 = np.zeros((padding_width, zommed_x.shape[1], zoomed_x.shape[2]), dtype=x.dtype)
            x_x = np.concatenate((zoomed_x,z0),axis=0)
        else:
            seed = random.randint(0,zoom_shape[0]-orig_shape[0])
            x_x = zoomed_x[seed:seed+origin_shape[0]][:][:]
            
        if zoom[1] < 1:
            padding_width=orig_shape[1]-zoom_shape[1]
            z0 = np.zeros((x_x.shape[1], padding_width,  x_x.shape[2]), dtype=x.dtype)
            x_y = np.concatenate((x_x,z0),axis=1)
        else:
            seed = random.randint(0,zoom_shape[1]-orig_shape[1])
            x_y = x_x[:][seed:seed+origin_shape[1]][:]
            
        if zoom[2] < 1:
            padding_width=orig_shape[2]-zoom_shape[2]
            z0 = np.zeros((x_y.shape[0], x_y.shape[1], padding_width), dtype=x.dtype)
            x_z = np.concatenate((x_y,z0),axis=0)
        else:
            seed = random.randint(0,zoom_shape[0]-orig_shape[0])
            x_z = x_y[seed:seed+origin_shape[0]][:][:]
            
        return x_z
            
    
    def __data_augmentation(self,list_IDs_temp):
        'perform data augmentation on batch of data loaded'
        #initialization
        X = np.empty((self.batch_size, *self.dim))
        shifted_X = np.empty((self.batch_size, *self.dim))
        rotated_X = np.empty((self.batch_size, *self.dim))
        shiftrotated_X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        #load data and perform augmentation
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')
            shifted_X[i] = random_shift(X[i])
            roated_X[i] = random_rotate(X[i])
            zoomed_X = random_zoom_and_crop(X[i])            

            # Store class
            y[i] = self.labels[ID]
            
        #concatenate augmented data together
        aug_X = np.concatenate((shifted_X, rotated_X, zoomed_X),axis=0)
        #expand dimension
        aug_X = np.expand_dims(aug_X, axis=1)
        #concatenate labels for augmented data
        aug_y = np.concatenate((y,y,y),axis=0)
        aug_Y = keras.utils.to_categorical(aug_y, num_classes=self.n_classes)
        return aug_X, aug_Y
        
