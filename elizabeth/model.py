import numpy as np
import tensorflow as tf
import os

from tensorflow._api.v2.image import rgb_to_grayscale

class Model:

    saved_speed_model = 'speed_model/'
    saved_angle_model = 'angle_model/'
    def __init__(self):
        self.speed_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))
        self.angle_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [100, 100])
        im=rgb_to_grayscale(im)
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        angles = np.arange(17)*5+50
        image = self.preprocess(image)
        
        pred_speed = self.speed_model.predict(image)[0]
        speed = (pred_speed[0]*35).astype(int)
        pred_angle = self.angle_model.predict(image)[0]
        angle = (pred_angle*80+50).astype(int)
        print('angle:', angle,'speed:', speed)
        
        return angle, speed
