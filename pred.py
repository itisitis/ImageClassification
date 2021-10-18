import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse


model = tf.keras.models.load_model("./training_1/saved_model/")
# path = args.input
def pred(args):
    path = args.input
    image = tf.keras.preprocessing.image.load_img(path,target_size=(380,380))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
#     plt.imshow(image)
    if predictions[0][0] > 0.8:
        print("outdoor")
        print("prob:", predictions[0][0]*100)
    elif predictions[0][0] < 0.2:
        print("indoor")
        print("prob:", predictions[0][0]*100)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input image')
    args = parser.parse_args()
    
    pred(args)
