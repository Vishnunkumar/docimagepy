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
    
  def __init__(self, scaler, model):
    self.scaler = scaler
    self.model = model

  def load_models(self):
    self.model = tf.keras.models.load_model('src/model.h5')
    self.scaler = joblib.load('src/sscaler.gz') 

  def Rotate_Image(self, scaler, model, imge):
    
    self.imge = imge
    img_width, img_height = 224, 224    
    rt_angle = 3
    while abs(rt_angle) > 2:
        imgex = cv2.resize(self.imge, (img_width, img_height))
        imgex = imgex.reshape(1, img_height, img_width, 3)
        imgex = imgex/255
        angle = self.model.predict(imgex)
        rt_angle = self.scaler.inverse_transform(angle)[0][0]
        print(rt_angle)
        im1 = Image.fromarray(self.imge)
        im2 = im1.rotate(-1*(rt_angle), PIL.Image.NEAREST, expand = 1, fillcolor = 'white')
        self.imge = np.array(im2)
        
    return im2, rt_angle


