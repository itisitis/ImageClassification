import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support


model = tf.keras.models.load_model("./training_1/saved_model/")
types = ('*.jpg', '*.png', '*.jpeg')

def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

def confusion(args):
    indoor_fol = args.class_1
    outdoor_fol = args.class_2
    indoor = []
    for files in types:
        indoor.extend(glob.glob(indoor_fol+files))
    outdoor = []
    for files in types:
        outdoor.extend(glob.glob(outdoor_fol+files))

    ind = [0]*len(indoor) 
    out = [1]*len(outdoor)

    data = indoor+outdoor
    label = ind+out

    pred = []
    for i, img in  enumerate(data):
        image = tf.keras.preprocessing.image.load_img(img,target_size=(380,380))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        if predictions[0][0] > 0.5:
            pred.append(1)
        elif predictions[0][0] < 0.5:
            pred.append(0)

    print(classification_report(pred, label))
    out = get_classification_report(pred, label)
    csv = args.out_path
    out.to_csv(csv + "evaluation.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_1', required=True, help='class-1')
    parser.add_argument('--class_2', required=True, help='class-2')
    parser.add_argument('--out_path', required=True, help='out_put_path')
    args = parser.parse_args() 
    confusion(args)
