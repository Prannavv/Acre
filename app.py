from sys import stderr

from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_mysqldb import MySQL
import mysql.connector


app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))
col=['stories_four','stories_one','stories_three','stories_two','lotsize','bedrooms','bathrms','driveway','recroom','fullbase','gashw','airco','garagepl','prefarea']

@app.route('/home.html')
def home2():
    return render_template('home.html')

@app.route('/index.html')
def hello_world():
    return render_template("index.html")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/result',methods=['GET','POST'])
def result():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="acre"
    )
    mycursor = mydb.cursor()
    if request.method == 'POST':
        data = request.form
        name = data['u']
        email = data['e']
        location = data['l']
        phoneno = data['n']

    mycursor.execute("insert into customer (name, email, location, phoneno) values(%s,%s,%s,%s)",(name, email, location, phoneno)) 
    mydb.commit()
    mycursor.close()  

    return render_template("result.html") 
       
@app.route('/about.html')
def about():
    return render_template("about.html")        


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction=model.predict(final)
    output=round(prediction[0],2)

    return render_template('index.html', pred='The predicted price of house is {} USD'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
