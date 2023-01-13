import tensorflow as tf
import os
from matplotlib import pyplot as plt
import keras
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
#--------------------------------------------------------------------- #
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" uncoment this to run on your CPU
#-------- tensorflow-gpu-2.10.0-----cuda v11.0.194-------cDNN v8.0-------Python 3.10.6#
#gpus = tf.config.experimental.list_physical_devices('GPU') uncoment this section to prevent  use too much memory from your gpu
#for gpu in gpus:
 #   tf.config.experimental.set_memory_growth(gpu,True)
#--------------------------------------------------------------------- #
data_dir = 'data' # the directory that haves N directories with the images

#loading with keras
data = tf.keras.utils.image_dataset_from_directory('data',batch_size=64,shuffle=True) # Keras creates a dataset with labels, resized images,batch size etc... 

class_names = data.class_names # get the names of eatch class  
print(class_names)

#pre-processing
data = data.map(lambda x,y:(x/255,y)) # normalize the image values ​​to improve performance.

trainSize = int(len(data)*.80) # 80% for training
validSize = int(len(data)*.20) # 20% for validation

train = data.take(trainSize)                       # take the batches for training
val = data.skip(trainSize).take(validSize)         # takes the batches for validation skipping the ones that were taken for training

# model configuration
model = tf.keras.models.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.15))
model.add(Flatten())
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam', metrics=['accuracy'],) #compila o modelo
model.summary() #shows the results of the model.

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5) #callback config for early stop the training
mc = tf.keras.callbacks.ModelCheckpoint(os.path.join('models','best_model-TEST.h5'), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)#callback config for model checkpoint

hist = model.fit(train, epochs=20, validation_data=val,verbose="auto",callbacks=[mc]) # training
model.save(os.path.join('models','imageclassifier-dataset-bom-45k.h5'))

# plot a graphic of the training results 
plt.plot(hist.history['loss'], label='Loss training data')
plt.plot(hist.history['val_loss'], label='Loss Validation Data')
plt.plot(hist.history['accuracy'], label='Accuracy training data')
plt.title('Model performance for 3D MNIST Keras Conv3D example')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# evaluate
loss,accuracy = model.evaluate(val) 
print(loss,accuracy)
