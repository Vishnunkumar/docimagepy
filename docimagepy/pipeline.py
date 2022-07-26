#This is importing the required libraries.
import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
from sklearn import preprocessing
import joblib
import re
import pytesseract

class DocImage:
"""
The above code is loading the model and scaler and then using the model to 
predict the angle of rotation of the image.
"""
  def __init__(self, scaler, model):
    self.scaler = scaler
    self.model = model

  
  def load_models(self):
  """
  Loading the model and scaler.
  """  
    self.model = tf.keras.models.load_model('src/model.h5')
    self.scaler = joblib.load('src/sscaler.gz') 

 
  def get_rotate_angle(self, scaler, model, imge):
  """
  The above code is used to rotate the image to the correct angle.
  """
    img_width, img_height = 224, 224    
    rt_angle = 10
    f_angle = 0
    while abs(rt_angle) > 8:
        imgex = cv2.resize(imge, (img_width, img_height))
        imgex = imgex.reshape(1, img_height, img_width, 3)
        imgex = imgex/255
        angle = loaded_model.predict(imgex)
        rt_angle = scaler.inverse_transform(angle)[0][0]
        rot_angle = -1*(rt_angle)
        im1 = Image.fromarray(imge)
        im2 = im1.rotate(rot_angle, PIL.Image.NEAREST, expand = 0, fillcolor = 'white')
        imge = np.array(im2)
        f_angle = f_angle + rot_angle

    return f_angle


