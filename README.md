# Deep-Cnn-Img-Classifier

This is a simplified version of Very Deep Convolutional Networks for Large-Scale Image Recognition(https://keras.io/api/applications/vgg/),
with the intention of being as simple as possible to use, to change the number of classes (types of images), you must  changed the last layer(model.add(tf.keras.layers.Dense(N, activation='softmax'))), where N are the number of classes you have. As an example, in my implementation there are 6 types of fish.



