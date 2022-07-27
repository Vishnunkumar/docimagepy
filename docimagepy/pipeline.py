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
  def __init__(self, imge):
    self.imge = imge
  
  def load_models(self):
    """
    Loading the model and scaler.
    """  
    
    self.model = tf.keras.models.load_model('src/model.h5')
    self.scaler = joblib.load('src/sscaler.gz') 

    return self.model, self.scaler

  def get_rotate_angle(self, scaler, model, imge, th_angle):

    """
    The above code is used to rotate the image to the correct angle.
    """
    self.th_angle = th_angle
    img_width, img_height = 224, 224    
    rt_angle = 10
    f_angle = 0
    while abs(rt_angle) > self.th_angle:
        imgex = cv2.resize(self.imge, (img_width, img_height))
        imgex = imgex.reshape(1, img_height, img_width, 3)
        imgex = imgex/255
        angle = self.model.predict(imgex)
        rt_angle = self.scaler.inverse_transform(angle)[0][0]
        rot_angle = -1*(rt_angle)
        im1 = Image.fromarray(self.imge)
        im2 = im1.rotate(rot_angle, PIL.Image.NEAREST, expand = 0, fillcolor = 'white')
        self.imge = np.array(im2)
        f_angle = f_angle + rot_angle

    return f_angle


