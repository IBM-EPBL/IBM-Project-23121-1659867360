import flask
from flask import request, render_template
from flask_cors import CORS
import joblib
 
app = flask.Flask(__name__, static_url_path='')
CORS(app)

@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predictEligibility():
    gre = int(request.form['GRE_Score'])
    toefl = int(request.form['TOEFEL_Score'])
    universityRating = int(request.form['u_rate'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = int(request.form['Research'])
    X = [[gre,toefl,universityRating,sop,lor,cgpa,research]]
    model = joblib.load('model.pkl')
    species = model.predict(X)[0]
    return render_template('predict.html',predict=species)
 
if __name__ == '__main__':
    app.run()
 

