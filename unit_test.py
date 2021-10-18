import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse


model = tf.keras.models.load_model("./training_1/saved_model/")
# path = args.input
def pred(args):
    indoor = args.indoor
    outdoor = args.outdoor
    image = tf.keras.preprocessing.image.load_img(indoor,target_size=(380,380))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    pred_in = model.predict(input_arr)
    
    image = tf.keras.preprocessing.image.load_img(outdoor,target_size=(380,380))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    pred_out = model.predict(input_arr)
#     "{:.2f}".format(a_flsoat)
    print(" indoor: ", "{:.2f}".format(100-(pred_in[0][0]*100)),"%")
    print(" outdoor: ", "{:.2f}".format(pred_out[0][0]*100),"%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indoor', required=True, help="indoor_image" )
    parser.add_argument('--outdoor', required=True, help="outdoor_image" )
    args = parser.parse_args()
    pred(args)
