# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:14:53 2018

@author: Jenario
"""

import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
#import base64
#import StringIO
import tensorflow as tf 
import numpy as np
#import cv2
import tensorflow as tf
import numpy as np
import os,glob
import sys,argparse
import pandas as pd
import numpy as np
import pickle
import json
import pyspark as sc
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow.python.platform import gfile
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
# Obtain the flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER='./'
def load_graph(trained_model):   
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph



@app.route('/',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        # Set Column Header Labels
        columns = ['freq', 'pwr', 'time', 'diff', 'pwrTime', "pDiff"]
        df = pd.read_csv(filename, parse_dates=['time'], header = None, names = columns)
        df[['pwr', 'freq', 'time', 'diff', 'pwrTime', "pDiff"]] = df[['pwr', 'freq', 'time', 'diff', 'pwrTime', "pDiff"]].apply(pd.to_numeric)
        df = df.dropna()
        
        image_size=128
        num_channels=3
        images = []
        N_TIME_STEPS = 200;
        N_FEATURES = 6;
        step = 1
        segments = []
        labels = []
        for i in range(0, len(df) - N_TIME_STEPS, step):
            fq = df['freq'].values[i: i + N_TIME_STEPS-1]
            pw = df['pwr'].values[i: i + N_TIME_STEPS-1]
            tm = df['time'].values[i: i + N_TIME_STEPS-1]
            diff = df['diff'].values[i: i + N_TIME_STEPS-1]
            pwt = df['pwrTime'].values[i: i + N_TIME_STEPS-1]
            pwd = df['pDiff'].values[i: i + N_TIME_STEPS-1]
            #label = stats.mode(df['type'][i: i + N_TIME_STEPS])[0][0]
            segments.append([fq, pw, tm, diff, pwt, pwd])
            #labels.append(label))
            
        data = df.as_matrix()
        data = data.reshape(1, 200, 6)
        # np.reshape(this_x,(1, FIN_SIZE))
        np.array(data).shape
        
        graph =app.graph
        
        y_pred = graph.get_tensor_by_name("y_:0")
          ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("input:0")
        #y_true = graph.get_tensor_by_name("y_true:0") 
        y_test_stream = np.zeros((1, 2))    
        sess= tf.Session(graph=graph)

        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        testing = {x: data}
        result=sess.run(y_pred, feed_dict = testing)
        # result is of this format [probabiliy_of_cats probability_of_dogs]
        #print()
        #pred=str(result[0][0]).split(" ")
        #print(pred)
        out={"jamm":str(result[0][0]),"normal":str(result[0][1]),"replay":str(result[0][2])}
        return jsonify(out)
        #return redirect(url_for('just_upload',pic=filename))

    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Running my first AI Demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">
          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">RF Signal Classification Demo</h1>
		            <h4 style="color:black">Upload new Signal </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>
    '''




app.graph=load_graph('./frozen_RF_Algo.pb')  
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)
    #app.run(host='localhost', port=int("9000"), debug=True, use_reloader=False)
    