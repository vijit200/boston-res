from flask import Flask,render_template,request
import pickle

import numpy as np
model = pickle.load(open('bos_res.pkl','rb'))
processing = pickle.load(open('process.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    if request.method == 'POST':
        CRIM = float(request.form['CRIM'])
        ZN = float(request.form['ZN'])
        INDUS = float(request.form['INDUS'])
        CHAS = float(request.form['CHAS'])
        NOX = float(request.form['NOX'])
        RM = float(request.form['RM'])
        AGE = float(request.form['AGE'])
        DIS = float(request.form['DIS'])
        RAD = float(request.form['RAD'])
        TAX = float(request.form['TAX'])
        PTRATIO = float(request.form['PTRATIO'])
        B = float(request.form['B'])
        LSTAT = float(request.form['LSTAT'])
        l =np.array([CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT])
        s = l.reshape(1,-1)
        scaling = processing.transform(s)
        model_eval = model.predict(scaling)[0]

        return render_template('index.html',predict_price = round(model_eval,2))


if __name__ == '__main__':
    app.run(debug=True)