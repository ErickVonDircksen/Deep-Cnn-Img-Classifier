import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.layers import  Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
#--------------------------------------------------------------------- #
# Erick H. Dircksen, Christian A. Carneiro, Raian A. Moretti 
#--------------------------------------------------------------------- #
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#-------- tensorflow v2.8.0-----cuda v11.0.194-------cDNN v8.0-------#
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
 #   tf.config.experimental.set_memory_growth(gpu,True)
#--------------------------------------------------------------------- #
data_dir = 'data'
#tratamento dos imgs

#loading com keras
data = tf.keras.utils.image_dataset_from_directory('data',batch_size=32,shuffle=True) # Keras cria um dataset com labels, resized images,batch size etc... 

class_names = data.class_names
print(class_names)

#pré porcessamento
data = data.map(lambda x,y:(x/255,y)) # normaliza os valores da imagem, para melhorar o desempenho.
# dataIterator = data.as_numpy_iterator() # cria um iterator pra podermos visualizar os batches de dados   

trainSize = int(len(data)*.80) # 80% para treinamento
validSize = int(len(data)*.20) # 20% para validação

train = data.take(trainSize)                         # pega os batches para treinaemento
val = data.skip(trainSize).take(validSize)         # pega os batches para validação pulandos os q foram pegos para treinamento 

# AI TIME!!!

model = tf.keras.models.Sequential() # inicia o modelo

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(2,2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer='adam', metrics=['accuracy'],) #compila o modelo
model.summary() #mostra os resultos do modelo.

hist = model.fit(train, epochs=15, validation_data=val,verbose="auto",) # treinamento 
model.save(os.path.join('models','imageclassifier-3-argumented.h5'))
# plota o resultado
plt.plot(hist.history['loss'], label='Loss training data')
plt.plot(hist.history['val_loss'], label='Loss Validation Data')
plt.plot(hist.history['accuracy'], label='Accuracy training data')
plt.title('Model performance for 3D MNIST Keras Conv3D example')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()




# evaluate
loss,accuracy = model.evaluate(val) # testa dois batches
print(loss,accuracy) #mostra os resultados do teste

#save the model
