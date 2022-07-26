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

    def Rotate_Image(self, scaler, model, imge):
        """
        The above code is rotating the image by a small angle and then predicting the angle of rotation. 
        This is done until the angle of rotation is less than 2 degrees.
        """
        self.imge = imge
        img_width, img_height = 224, 224    
        rt_angle = 10
        while abs(rt_angle) > 5:
            imgex = cv2.resize(self.imge, (img_width, img_height))
            imgex = imgex.reshape(1, img_height, img_width, 3)
            imgex = imgex/255
            angle = self.model.predict(imgex)
            rt_angle = self.scaler.inverse_transform(angle)[0][0]
            im1 = Image.fromarray(self.imge)
            im2 = im1.rotate(-1*(rt_angle), PIL.Image.NEAREST, expand = 1, fillcolor = 'white')
            self.imge = np.array(im2)
            
        return im2, rt_angle




