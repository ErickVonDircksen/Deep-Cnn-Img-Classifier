import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.preprocessing import image
from PIL import Image
import visualkeras

def prepare_image (file):
    im_resized = tf.keras.preprocessing.image.load_img(file, target_size = (256,256))
    img_array = tf.keras.utils.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return image_array_expanded/255

dict = ['Bagre', 'Carpa Capim', 'Chanda Vitreo', 'Goby Negro', 'Knifefish', 'Tilapia']

names = ['Bagre', 'Carpa Capim', 'Chanda Vitreo', 'Goby Negro', 'Knifefish', 'Tilapia','Goby Negro','Tilapia-render','Knifefish','Bagre-zoom','Carpa-Minecraft','Bogdan']

model = keras.models.load_model('D:\Area de Trabalho\AI-Keras\models\imageclassifier-dataset-bom-45k.h5')
visualkeras.layered_view(model, legend=True) 
visualkeras.layered_view(model,legend=True,scale_xy=1, scale_z=1, max_z=30, to_file='output.png') # write and show
for x in range(12):
    testIMG = 'testIMG'+str(x)+'.jpg'
    print(testIMG)
    img_array = prepare_image(testIMG)

    predictions = model.predict(img_array)
    print(predictions)

    print(
        names[x]+ " most likely belongs to {} with a {:.2f} percent confidence."
        .format(dict[np.argmax(predictions)], 100 * np.max(predictions))

    )

