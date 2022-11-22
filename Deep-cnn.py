import tensorflow as tf
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#-------- tensorflow v2.8.0-----cuda v11.0.194-------cDNN v8.0-------#
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
 #   tf.config.experimental.set_memory_growth(gpu,True)
#--------------------------------------------------------------------- #
data_dir = 'data'
#tratamento dos imgs

#loading com keras
data = tf.keras.utils.image_dataset_from_directory('data',batch_size=16,shuffle=True) # Keras cria um dataset com labels, resized images,batch size etc... 

#pré porcessamento
data = data.map(lambda x,y:(x/255,y)) # normaliza os valores da imagem, para melhorar o desempenho.
dataIterator = data.as_numpy_iterator() # cria im iterator pra podermos visualizar os batches de dados
batch = dataIterator.next()             # mostra um batch 
#print(batch[1])                        # batch[1]são os labels, batch[0]são as imagens, cada nº de label representa uma classe(tipo de peixe no nosso caso).

#  divide as imagens em categorias, TREINAMENTO E TESTAGEM, devem estar embaralhas neste ponto
#print(len(data)) # numero de batches 

trainSize = int(len(data)*.7)   # 70% para treinamento
validSize = int(len(data)*.2)+1 # 20% para validação
testSize  = int(len(data)*.1)+1 # 10% to evaluation 

#print(trainSize+validSize+testSize) # confirma se os batches foram divididos corretamente

train = data.take(trainSize)                         # pega os batches para treinaemento
val  =  data.skip(trainSize).take(validSize)         # pega os batches para validadção pulandos os q foram pegos para treinamento 
test =  data.skip(trainSize+validSize).take(testSize)# o mesmo para os batches de teste


# AI TIME!!!

model = tf.keras.models.Sequential() # inicia o modelo
#adiciona camadas ao modelo;
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) #compila o modelo
model.summary() #mostra os resultos do modelo.

logdir='logs'   #cria logs pra fine tunning do moledo
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val,verbose="auto") # treinamento 

# plota o resultado
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['accuracy'], color='green', label='accu')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()



# evaluete
pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print('\n')
print(pre.result(), re.result(), acc.result())

#save the model
model.save(os.path.join('models','imageclassifier.h5'))
