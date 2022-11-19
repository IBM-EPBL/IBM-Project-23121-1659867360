import flask
from flask import request, render_template
from flask_cors import CORS
import requests
from flask_mysqldb import MySQL

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "JyhzzPTqYb5vdnL-g25mEurwEUgEHsrht4XjZo6UvyUm"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = flask.Flask(__name__, static_url_path='')
CORS(app)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'contactdata'

mysql = MySQL(app)

@app.route('/redirect', methods=['POST'])
def sendThanksPage():
     firstName = request.form['firstname']
     lastName = request.form['lastname']
     phone = request.form['phone']
     email = request.form['email']
     message = request.form['message']
     cursor = mysql.connection.cursor()
     cursor.execute(''' INSERT INTO contactdata VALUES(%s,%s,%s,%s,%s)''',(firstName,lastName,phone,email,message))
     mysql.connection.commit()
     cursor.close()
     return f"Thank You For Contacting Us "
     
@app.route('/about')
def sendAboutPage():
    return render_template('about.html')

@app.route('/feedback')
def sendFeedbackPage():
    return render_template('index.html')

@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    gre = int(request.form['GRE_Score'])
    toefl = int(request.form['TOEFEL_Score'])
    universityRating = int(request.form['u_rate'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = int(request.form['Research'])
    X = [[gre,toefl,universityRating,sop,lor,cgpa,research]]

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": ['gre','toefl','universityRating','sop','lor','cgpa','research'], "values": X}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/0bfdeda4-f2a5-45c8-9262-ad4eb310e85f/predictions?version=2022-11-06', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print(response_scoring)
    predictions = response_scoring.json()
    predict = predictions['predictions'][0]['values'][0][0]
    print("Final prediction :",predict)

    
    # showing the prediction results in a UI# showing the prediction results in a UI
    return render_template('predict.html', predict=predict)

if __name__ == '__main__' :
    app.run(debug= False)
  
