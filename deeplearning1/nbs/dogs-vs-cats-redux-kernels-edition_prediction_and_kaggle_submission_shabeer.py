
# coding: utf-8

# # Apply vgg16 model and predict class for test data of https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition and submit prediction results to Kaggle.

# In[1]:

# Module versions
import sys
import keras
import theano
import numpy
import pandas

print("Python version:" + sys.version)
print("Keras version:" + keras.__version__)
print("Theano version:" + theano.__version__)
print("Numpy version:" + numpy.__version__)


# ## Setup Keras and its backend before using it

# In[2]:

print("Keras backend:" + keras.backend.backend())

print("Keras backend image data format:Before:" + keras.backend.image_data_format())
# change image_data_format to channels_first
keras.backend.set_image_data_format('channels_first')
print("Keras backend image data format:After:" + keras.backend.image_data_format())
print("Keras backend image_dim_ordering:After:" + keras.backend.image_dim_ordering())

print("Keras backend epsilon:Before:" + str(keras.backend.epsilon()))
# change epsilon to 1e-7
keras.backend.set_epsilon(1e-7)
print("Keras backend epsilon:After:" + str(keras.backend.epsilon()))

print("Keras backend floatx:Before:" + str(keras.backend.floatx()))
# change floatx to float32
keras.backend.set_floatx('float32')
print("Keras backend floatx:After:" + str(keras.backend.floatx()))


# In[3]:

import utils; reload(utils)
import vgg16; reload(vgg16)
#get_ipython().magic(u'matplotlib inline')
#from utils import plots
from vgg16 import Vgg16


# ### Training, Validation and Testing data path

# #### Enable either floydhub or local path

# In[4]:

floyd_or_local = "floyd"
dataset = "global" # sample or global


# In[21]:

#path containing sample, train, valid and test1 directries
global_path = ""
# trimmed down version of train,validation and testing data from above path
sample_path = ""
# path to save any artifacts created
output_path = ""

if floyd_or_local == "local":
    global_path = "data/dogscats/"
    sample_path = "data/dogscats/sample/"
    output_path = "./"
    utils.set_keras_cache_dir("/home/shabeer/.keras/")
    
    # Prepare local global/sample test path
    # Ignore errors if this notebook is run in non-local environment.
    get_ipython().system(u'global_path="data/dogscats/"')
    get_ipython().system(u'mkdir -p $global_path/test1/unknown/')
    get_ipython().system(u'mv $global_path/test1/*.jpg $global_path/test1/unknown/')

    get_ipython().system(u'sample_path="data/dogscats/sample/"')
    get_ipython().system(u'mkdir -p $sample_path/test1/unknown/')
    get_ipython().system(u'cp $global_path/test1/unknown/4*09.jpg $sample_path/test1/unknown/    ')
    
else:
    global_path = "/input/dogscats/"
    sample_path = "/input/dogscats/sample/"
    output_path = "/output/"
    utils.set_keras_cache_dir("/input/models/")    

if dataset == "sample":
    path = sample_path
else:
    path = global_path

test_path = global_path + "/test1/"    


# ### Create vgg16 model with its weights loaded

# In[6]:

vgg = Vgg16()


# In[7]:

# Based on memory available, choosing a medium value. Max could be 64, above which could be - out of memory
batch_size = 16
train_batches = vgg.get_batches(path + '/train/', batch_size = batch_size, class_mode='categorical')
validation_batches = vgg.get_batches(path + '/valid/', batch_size = batch_size)
#test_batches = vgg.get_batches(path + '../test1/', batch_size = batch_size * 4)


# In[8]:

print("Number of classes in vgg model before fine tuning:" + str(len(vgg.classes)))
# fine tune vgg16 model to 2 classes - dogs and cats
vgg.finetune(train_batches)
print("Number of classes in vgg model before after tuning:" + str(len(vgg.classes)))
print("Classes after tuning:" + str(vgg.classes))


# ## TODO train and validate vgg16 model to 2 classes data.

# In[9]:

import datetime
time_before_starting_training = datetime.datetime.now()
print(time_before_starting_training)

# train & validate
vgg.fit(batches= train_batches, val_batches= validation_batches, nb_epoch=1)
time_after_training =  datetime.datetime.now() 
print(time_after_training)

train_imgs, train_labels = next(train_batches)
print("train_imgs shape:" + str(train_imgs.shape))

print("Time taken to train & validate: " + str(time_after_training - time_before_starting_training))


# ## Save model and weights

# In[10]:

print("Saving model configuration.")
model_json = vgg.model.to_json()
#print(model_json)
with open(output_path + '/model.json', 'w') as f:
    f.write(model_json)
print("Saved model configuration.")
    
#serialize weights to hdf5
model_weights = vgg.model.save_weights(output_path + '/model_weights.h5')
print("Saved model weights.")


# In[11]:

#?vgg.model.save_weights


# ## Test and predict labels

# In[22]:

test_batches = vgg.get_batches(test_path, batch_size = 8, class_mode=None)
imgs = next(test_batches)
print(imgs.shape)
preds, idxs, classes = vgg.predict(imgs)


# In[23]:

#plots(imgs[0:11])
print(preds[0:11])
print(idxs[0:11])
print(classes[0:11])


# In[14]:

#?vgg.model.predict_generator


# In[24]:

time_before_starting_testing = datetime.datetime.now()
print(time_before_starting_testing)
batch_size = 8

# both test_batches and predictions below are generators
#test_batches, predictions = vgg.test(test_path, batch_size = 8)
test_batches = vgg.get_batches(test_path, batch_size=batch_size, class_mode =None)
print("List of images across all test_batches: " + str(test_batches.samples))

import math
steps = int(math.ceil(test_batches.samples*1.0/batch_size))
print("Number of steps:" + str(steps))

predictions  = vgg.model.predict_generator(test_batches, steps = steps, verbose=1)

time_after_testing = datetime.datetime.now()
print(time_after_testing)

time_taken = time_after_testing - time_before_starting_training

print("Time taken to test: " + str(time_taken))

print(predictions.shape)
print(predictions[0:11])
print("Probability of image being a dog:" + str(predictions[0:11,1]))


# ## Construct Kaggle submission

# In[34]:

print(vgg.classes)
idx = numpy.argmax(predictions, axis = 1) # idx within each row of predictions, which contains max probability.
print(idx[0:11])
classes_predicted = map(lambda i: vgg.classes[i], idx)
print(classes_predicted[0:11])

from pandas import Series
from pandas import DataFrame

print(test_batches.filenames[0:11])
filenames = map(lambda f: f.replace('unknown/', '').replace('.jpg', ''), test_batches.filenames)
print(filenames[0:11])

# probability of image being a dog ( 1=dog, 0=cat)
dog_prob = [str("%.12f" % p) for p in predictions[:,1]]

p = pandas.concat([Series(filenames), Series(dog_prob)], axis = 1, keys = ['id', 'label'])
print(p[0:11])
p.to_csv(output_path + '/dogs-vs-cats-redux-kernels-edition_predictions_shabeer.csv', header= True, mode='w', index=False)

