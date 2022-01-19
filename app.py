from flask_cors import CORS
from flask import Flask,request,jsonify
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import open
from keras.applications import vgg16
from keras.layers import Dense
from keras.preprocessing import image
app=Flask(__name__)
CORS(app)

@app.route('/hello', methods=['POST'])
def hello():
    if request.method == 'POST':
        uploaded_file = request.files
        print(uploaded_file['data'].save(uploaded_file['data'].filename))
        vgg_model=keras.models.load_model("./vgg_model_brain.h5")
        arr=np.asarray(open(uploaded_file['data'].filename).resize((224,224)))
        brain_img=arr.astype(np.float32)
        img_array=image.img_to_array(brain_img)
        img_list=[]
        img_list.append(keras.applications.vgg16.preprocess_input(img_array))
        X_test=np.array(img_list)
        predict_x=vgg_model.predict(X_test) 
        # y_pred = vgg_model.predict_classes(X_test)
        print(predict_x[0][0])
        os.remove(uploaded_file['data'].filename)
        res=dict()
        res['prob']=str(predict_x[0][0])
    return res

@app.route("/test")
def test():
    return "Test successful"
    
if __name__=="__main__":
    app.run(debug=True)