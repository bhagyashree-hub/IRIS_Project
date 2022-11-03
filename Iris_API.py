from distutils.log import debug
from flask import Flask ,render_template,request,jsonify
import numpy as np
import pickle
from Iris_Project_Data.utils import IrisSpeciesPrediction
import config


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/predict_spieces', methods = ['POST'])
def final_predict_spieces():
    SepalLengthCm = float(request.form["SepalLengthCm"])
    SepalWidthCm =float(request.form["SepalWidthCm"]) 
    PetalLengthCm = float(request.form["PetalLengthCm"])
    PetalWidthCm = float(request.form["PetalWidthCm"])

    Obj_Species = IrisSpeciesPrediction(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    prediction = Obj_Species.get_predict_spieces()

    return render_template("after.html",prediction=prediction)

if __name__ == "__main__":
    app.run(host ="0.0.0.0",port=8080,debug=True)