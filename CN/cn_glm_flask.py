from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import secure_filename, redirect
import joblib
import pandas as pd
import os
import glob
import numpy as np
from sksurv.metrics import concordance_index_censored

UPLOAD_FOLDER = 'upload_folder'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = joblib.load(filename='final_models/cn_glm_final.joblib')

list_of_files = glob.glob('upload_folder/*.csv')


@app.route('/')
def home():
    return render_template('template/home.html')


if list_of_files:
    test_dat = max(list_of_files, key=os.path.getctime)

app = Flask(__name__)

@app.route('/')
def index():
    pred = model.predict(test_dat.iloc[:, 2:])
    return render_template('template/index.html', pred=str(pred))

if __name__ == '__main__':
    app.run()
