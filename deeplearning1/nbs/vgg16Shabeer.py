from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import ZeroPadding2D,Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda,Flatten,Dense,Dropout
from keras.utils.data_utils import get_file

import numpy as np
import utils
import json

def vgg_preprocess(x):
    """
    Mean value of RGB value
    """
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32).reshape(3,1,1) # 3 is number of channels
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb -> bgr


'''
Re-creation of vgg16 imagenet model.
For documentation refer, vgg16.py
'''
class Vgg16Shabeer():
   
    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.get_classes()
        self.create()
    
    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir = 'models/', cache_dir = utils.get_keras_cache_dir())
        with open(fpath) as f:
            class_dict = json.load(f)
            
        #class_dict looks like {"0": ["n01440764", "tench"], 
        #                       "1": ["n01443537", "goldfish"], 
        #                       ....}
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
    
    def get_batches(self, path, batch_size, gen = ImageDataGenerator(), class_mode = 'categorical', shuffle = True):
        return gen.flow_from_directory(path, target_size = (224,224), batch_size = batch_size, class_mode = class_mode, shuffle = shuffle)
        
    def ConvBlock(self, layers, filters):
        model = self.model
        
        for i in range(layers):
            model.add(ZeroPadding2D(padding=(1,1)))
            model.add(Conv2D(filters, kernel_size = (3,3), activation = 'relu'))
            
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
        
    def FCBlock(self):
        model = self.model
        
        model.add(Dense(4096, activation ='relu'))
        model.add(Dropout(rate = 0.5))
    
    def create(self):
        model = self.model = Sequential()
        model.add(Lambda(function = vgg_preprocess, input_shape=(3,224,224), output_shape = (3, 224, 224)))
        
        self.ConvBlock(2,64)
        self.ConvBlock(2,128)
        self.ConvBlock(3,256)
        self.ConvBlock(3,512)
        self.ConvBlock(3,512)     
        
        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(units = 1000, activation = 'softmax'))
        
        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, origin = self.FILE_PATH+fname , cache_subdir='models', cache_dir = utils.get_keras_cache_dir()))
        
    def compile(self, lr = 0.001):
        self.model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
    def ft(self,num):
        model = self.model
        
        # Remove last layer, which has 1000 output
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        
        # Add a dense layer with number of outputs matching input parameter - num
        model.add(Dense(num, activation = 'softmax'))
        
        # now compile the model, to apply the changes done.
        self.compile()
        
        
    def finetune(self, batches):
        self.ft(batches.num_class)
        classes = list(iter(batches.class_indices))
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}
        
        # sort 
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
            
        self.classes = classes
    
    def fit_data(self, trn, labels, val, val_labels, nb_epoch =1, batch_size=64):
        self.model.fit(x=trn, y= labels, validation_data = (val,val_labels), epochs = nb_epoch, batch_size = batch_size)
    
    def fit(self, batches, val_batches, nb_epoch = 1):
        self.model.fit_generator(batches, steps_per_epoch = batches.samples, epochs = nb_epoch, validation_data = val_batches, validation_steps = val_batches.samples)
    
    def predict(self, imgs, details = False):
        # predict probability of each class
        all_predictions = self.model.predict(imgs)
        
        # get index of highest probability
        idxs = np.argmax(all_predictions, axis=1)
        
        # get values of highest probability
        preds = [all_predictions[i, idxs[i]] for i in range(len(idxs))]
        
        # get class label corresponding to highest probability
        classes = [self.classes[idx] for idx in idxs]
        return np.array(all_predictions), idxs, classes
    
    def test(self, path, batch_size = 8):
        test_batches = get_batches(path, batch_size = batch_size, shuffle=False, class_mode = None)
        return test_batches, self.model.predict_generator(test_batches, steps = test_batches.samples)