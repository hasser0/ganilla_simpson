from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy, mse, mae
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, Input, MaxPool2D, Conv2DTranspose, UpSampling2D, Add, LeakyReLU, Flatten, Dense
from keras_contrib.layers import InstanceNormalization
from keras.utils.layer_utils import count_params
from sklearn.utils import shuffle
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
import os

PATH_TO_MODELS = "./models/run3"
DATASET = "human_simpson"
#create to begin from 0
mode = "create"
batch_size = 10
prev_epoch = 0
epochs = 1002

def create_generator():
  def residual_block(input_layer, filters, kernel_size=(3,3), final_strides=1):
    layer = input_layer
    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(layer)
    layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
    layer = ReLU()(layer)
    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same")(layer)
    layer = InstanceNormalization(axis=-1, center=False, scale=False)(layer)
    layer = Concatenate()([input_layer,layer])
    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=final_strides, padding="same")(layer)
    layer = ReLU()(layer)
    return layer

  #INPUT
  input_layer = Input(shape=(256,256,3))
  layer = Conv2D(filters=32, kernel_size=7, padding="same")(input_layer)
  layer = InstanceNormalization(axis=-1)(layer)
  layer = ReLU()(layer)

  #LAYER1
  l1b1 = residual_block(layer, filters=64)
  l1b2 = residual_block(l1b1, filters=64, final_strides=2)

  #LAYER2
  l2b1 = residual_block(l1b2, filters=64)
  l2b2 = residual_block(l2b1, filters=64, final_strides=2)

  #LAYER3
  l3b1 = residual_block(l2b2, filters=128)
  l3b2 = residual_block(l3b1, filters=128, final_strides=2)

  #LAYER4
  l4b1 = residual_block(l3b2, filters=256)
  l4b2 = residual_block(l4b1, filters=256, final_strides=2)

  #UPSAMPLING1
  up1 = Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, padding="same")(l4b2)
  up1 = Concatenate()([up1, l3b2])
  up1 = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(up1)


  #UPSAMPLING2
  up2 = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")(up1)
  up2 = Concatenate()([up2, l2b2])
  up2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(up2)

  #UPSAMPLING3
  up3 = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")(up2)
  up3 = Concatenate()([up3, l1b2])
  up3 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(up3)

  #UPSAMPLING4
  up4 = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=2, padding="same")(up3)
  up4 = Conv2DTranspose(filters=3, kernel_size=7, strides=1, padding="same")(up4)

  model = Model(inputs=input_layer, outputs=up4)
  return model

def create_discriminator():
  input_layer = Input(shape=(256,256,3))
  layer = Conv2D(filters=32, kernel_size=4, strides=2, padding="same")(input_layer)
  layer = LeakyReLU(0.2)(layer)
  layer = Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(layer)
  layer = InstanceNormalization(axis=-1)(layer)
  layer = LeakyReLU(0.2)(layer)
  layer = Conv2D(filters=128, kernel_size=4, strides=2, padding="same")(layer)
  layer = InstanceNormalization(axis=-1)(layer)
  layer = LeakyReLU(0.2)(layer)
  layer = Conv2D(filters=256, kernel_size=4, strides=1, padding="same")(layer)
  layer = InstanceNormalization(axis=-1)(layer)
  layer = LeakyReLU(0.2)(layer)
  output_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(layer)
  model = Model(inputs=input_layer, outputs=output_layer)
  return model

train_size = batch_size
trainA_flow = ImageDataGenerator()
trainA = trainA_flow.flow_from_directory(directory=f"./datasets/{DATASET}",
                                classes=["trainA"],
                                batch_size=train_size)
trainB_flow = ImageDataGenerator()
trainB = trainB_flow.flow_from_directory(directory=f"./datasets/{DATASET}",
                                classes=["trainB"],
                                batch_size=train_size)

#CREATE MODELS
if mode == "create":
  g_AB = create_generator()
  g_BA = create_generator()
  d_A = create_discriminator()
  d_B = create_discriminator()
  d_A.compile(optimizer=Adam(learning_rate=0.0002), loss=mse, metrics=["accuracy"])
  d_B.compile(optimizer=Adam(learning_rate=0.0002), loss=mse, metrics=["accuracy"])
else:
  g_AB = load_model(os.path.join(PATH_TO_MODELS, f"g_AB_{prev_epoch}.h5"), custom_objects={"InstanceNormalization":InstanceNormalization})
  g_BA = load_model(os.path.join(PATH_TO_MODELS, f"g_BA_{prev_epoch}.h5"), custom_objects={"InstanceNormalization":InstanceNormalization})
  d_A = load_model(os.path.join(PATH_TO_MODELS, f"d_A_{prev_epoch}.h5"), custom_objects={"InstanceNormalization":InstanceNormalization})
  d_B = load_model(os.path.join(PATH_TO_MODELS, f"d_B_{prev_epoch}.h5"), custom_objects={"InstanceNormalization":InstanceNormalization})

imgA = Input(shape=(256,256,3))
imgB = Input(shape=(256,256,3))
fakeA = g_BA(imgB)
fakeB = g_AB(imgA)
reconsA = g_BA(fakeB)
reconsB = g_AB(fakeA)
idA = g_BA(imgA)
idB = g_BA(imgB)
#validA = d_A(g_BA(imageB))
validA = d_A(fakeA)
#validB = d_B(g_AB(imageA))
validB = d_B(fakeB)
combined = Model(inputs=[imgA, imgB], outputs=[validB, validA, reconsA, reconsB, idA, idB])
combined.compile(loss=[mse, mse, mae, mae, mae, mae], loss_weights=[1000, 1000,1,1,1,1])


for epoch in range(prev_epoch, epochs):
  print(f"\nEpoch {epoch}: {str(datetime.now())}")
  imageA, _ = next(trainA)
  imageB, _ = next(trainB)
  predictA_from_B = g_BA.predict(imageB)
  predictB_from_A = g_AB.predict(imageA)
  d_A.trainable = True
  d_B.trainable = True

  #TRAIN D_A
  ones = np.ones((train_size,32,32,1))
  zeros = np.zeros((train_size,32,32,1))
  y_train = np.concatenate([ones, zeros],axis=0)
  X_train = np.concatenate([imageA, predictA_from_B], axis=0)
  X_train, y_train = shuffle(X_train, y_train)
  lossesA = d_A.train_on_batch(X_train, y_train)
  print(f"D_A losses: {lossesA}")
  print("D-A trained")
  #TRAIN D_B
  ones = np.ones((train_size,32,32,1))
  zeros = np.zeros((train_size,32,32,1))
  y_train = np.concatenate([ones, zeros],axis=0)
  X_train = np.concatenate([imageB, predictB_from_A], axis=0)
  X_train, y_train = shuffle(X_train, y_train)
  lossesB = d_B.train_on_batch(X_train, y_train)
  print(f"D_B losses: {lossesB}")
  print("D-B trained")
  #TRAIN COMBINED MODEL
  d_A.trainable = False
  d_B.trainable = False

  all_real = np.ones((train_size,32,32,1))
  lossesC = combined.train_on_batch([imageA, imageB], [all_real, all_real, imageA, imageB, imageA, imageB])
  print(f"Generator losses: {lossesC}")
  print("Generators trained")
  with open("losses.csv","a") as csv:
    line = ",".join([str(item) for item in [epoch]+lossesA+lossesB+lossesC])+"\n"
    csv.write(line)
  print("Losses saved")
  if epoch%100 == 0:
    print("Saving models")
    #fakeB = g_AB.predict(imageA[0].reshape((1,256,256,3)))
    #fakeA = g_BA.predict(imageB[0].reshape((1,256,256,3)))
    #cv2.imwrite(os.path.join(PATH_TO_IMAGES, f"fakeA_{epoch}.png"),fakeA.reshape((256,256,3)))
    #cv2.imwrite(os.path.join(PATH_TO_IMAGES, f"fakeB_{epoch}.png"),fakeB.reshape((256,256,3)))
    g_AB.save(os.path.join(PATH_TO_MODELS, f"g_AB_{epoch}.h5"))
    g_BA.save(os.path.join(PATH_TO_MODELS, f"g_BA_{epoch}.h5"))
    d_A.save(os.path.join(PATH_TO_MODELS, f"d_A_{epoch}.h5"))
    d_B.save(os.path.join(PATH_TO_MODELS, f"d_B_{epoch}.h5"))

