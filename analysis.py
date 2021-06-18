from keras.applications.vgg16 import VGG16
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# import PIL
import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential


model = VGG16(weights = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# model = tf.keras.models.load_model('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()
#Create a Dataframes with defined columns
column_names = ["frame", "items",]
mainData = pd.DataFrame(columns= column_names)

def searchDataFrame(query):
  indexList = mainData['items'].str.contains(query)
  arry = []
  arry = mainData[indexList]['frame']
  print(arry)
  return arry

def addToDataFrame(frame,items,):
  print("adding to Dataframe")
  new_row = {'frame':frame, 'items':items,}
  global mainData
  mainData = mainData.append(new_row, ignore_index=True)
  print(mainData)

# Function to convert  
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele
        str1 += " , "  
    
    # return string  
    return str1


def analyzeFrame(frame):
  # load an image from file
  image = load_img('static/'+frame, target_size=(224, 224))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the VGG model
  image = preprocess_input(image)
  # predict the probability across all output classes
  yhat = model.predict(image)
  # convert the probabilities to class labels
  label = decode_predictions(yhat)
  # retrieve the most likely result, e.g. highest probability
  label3 = label[0]
  label4 = label3[:4]
  label2 = []
  for item in label4:
    label2.append(item[1])
  itemString = listToString(label2)
  # print the classification
  # print('%s (%.2f%%)' % (label2[1], label2[2]*100))
  print(label2[1][1])
  addToDataFrame(frame,itemString)
  return label2


def createDataFrame():
  print("Dataframe Created")


def analyzeVideo():
  directory = "Frames"
  for filename in os.listdir(directory):
      if filename.endswith(".jpg"): 
          analyzeFrame(filename)
          continue
      else:
          continue

def convertToFrames(location):
    vidcap = cv2.VideoCapture(location)
    success,image = vidcap.read()
    print(success)
    count = 0
    while success:
        cv2.imwrite("Frames/frame%d.jpg" % count, image)
        # analyzeFrame("Frames/frame%d.jpg" % count)
            # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return True

def newConvertToFrames(location):
    import os
    import glob

    files = glob.glob('static/Frames/*')
    for f in files:
        os.remove(f)
    vid = cv2.VideoCapture(location)

    index = 0
    while(True):
        ret, frame = vid.read()
        if not ret: 
            break
       
        if index%10==0:
            cv2.imwrite("static/Frames/frame%d.jpg" % index, frame)
            analyzeFrame("Frames/frame%d.jpg" % index)
            print('Read a new frame: ', index)
        index += 1
    return True
