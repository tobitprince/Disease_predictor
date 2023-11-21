from flask import Flask, render_template, request, redirect, Markup, url_for, session, flash
import numpy as np
import pandas as pd
from disease_dic import disease_dic
import requests
import config
import json
import pickle
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from model import ResNet9
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re, hashlib
from requests.auth import HTTPBasicAuth
from datetime import datetime
from dotenv import load_dotenv
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import base64



app = Flask(__name__)
load_dotenv()

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = os.getenv('app.secret_key')

# access token
consumerKey = os.getenv('consumerKey') #Fill with your app Consumer Key
consumerSecret = os.getenv('consumerSecret') # Fill with your app Secret
base_url = os.getenv('base_url')

# Enter your database connection details below
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')


# Intialize MySQL
mysql = MySQL(app)

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   '(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_model_path = 'Trained_Model/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


crop_recommendation_model_path = 'Trained_Model/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



@ app.route('/')
def home1():
    return render_template('index2.html')

@ app.route('/home2')
def home2():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('index3.html', username=session['username'],title="Home")
    # User is not loggedin redirect to login page
    return redirect(url_for('login')) 
@ app.route('/home3')
def home3():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('admin/home.html', username=session['username'],title="Home")
    # User is not loggedin redirect to login page
    return redirect(url_for('admin')) 
@ app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')

@ app.route('/login')
def login():
    print("mysql.connection:", mysql.connection)  # Print the value of mysql.connection
    return render_template('login/signup-login.html')

# http://localhost:5000/login/ - this will be the login page, we need to use both GET and POST requests
@app.route('/logini', methods=['GET', 'POST'])
def logini():
    # Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        # Check database connection
        # Create variables for easy access
        email = request.form['email']
        password = request.form['password']

        # Retrieve the hashed password
        hash = password + app.secret_key
        hash = hashlib.sha1(hash.encode())
        password = hash.hexdigest()

        # Get cursor
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        # Check if account exists using MySQL
        cursor.execute('SELECT * FROM farmers WHERE email = %s AND password = %s', (email, password))

        # Fetch one record and return result
        account = cursor.fetchone()

        # If account exists in accounts table in our database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']

            # Commit changes to the database
            mysql.connection.commit()

            # Redirect to home page
            return redirect(url_for('home2'))
        else:
            # Account doesn't exist or username/password incorrect
            msg = 'Incorrect username/password!'
            print("Flash message set!")
            flash("Incorrect username/password!", "danger")
    print("Redirecting to login page")
    return render_template('login/signup-login.html', msg=msg)
# http://localhost:5000/pythonlogin/register 
# This will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'uname' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['uname']
        password = request.form['password']
        email = request.form['email']
        status = 0
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM farmers WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM farmers WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
            # Hash the password
            hash = password + app.secret_key
            hash = hashlib.sha1(hash.encode())
            password = hash.hexdigest()
            print(password)
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO farmers VALUES (NULL, %s, %s, %s, %s)', (username,email, password, status))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            flash("You have successfully registered!", "success")
            return render_template('login/signup-login.html', msg=msg)

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('login/signup-login.html', msg = msg)

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction)

        else:

            return render_template('try_again.html')

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img = file.read()

        prediction = predict_image(img)

        prediction = Markup(str(disease_dic[prediction]))
        return render_template('disease-result.html', prediction=prediction)

    return render_template('disease.html')

    @app.route('/service-worker.js')
    def sw():
        return app.send_static_file('service-worker.js')
    
# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for logged in users
@app.route('/profile')
def profile():
    # Check if the user is logged in
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM farmers WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('login/profile.html', account=account)
    # User is not logged in redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


#####MPESAAA

@ app.route('/mpesa')
def mpesa():
    return render_template('mpesa/index.html')

#intitate mpesa request
@app.route('/pay')
def pay():
      if request.method == 'POST':
            amount = request.form['amount']
            phone = request.form['phone']

            endpoint = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest'
            access_token = get_access_token()
            headers = { "Authorization": "Bearer %s" % access_token }
            Timestamp = datetime.now()
            times = Timestamp.strftime("%Y%m%d%H%M%S")
            password = "174379" + "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919" + times
            password = base64.b64encode(password.encode['utf-8'])

            data = {
                "BusinessShortCode": "174379",
                "Password": password,
                "Timestamp": times,
                "TransactionType": "CustomerPayBillOnline",
                "PartyA": phone, # fill with your phone number
                "PartyB": "174379",
                "PhoneNumber": phone, # fill with your phone number
                "CallBackURL": os.getenv('my_endpoint') + "/lnmo-callback",
                "AccountReference": "TestPay",
                "TransactionDesc": "HelloTest",
                "Amount": amount
            }

            res = requests.post(endpoint, json = data, headers = headers)
            return res.json()
      elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = "Please fill out the form!", "danger"
        # Show mpesa form with message (if any)
        return render_template('mpesa/index.html', msg = msg)
      
#consume mpesa callback
@app.route('/lnmo-callback', methods = ["POST"])
def incoming():
      data = request.get_json()
      print(data)
      return "ok"
      
   
@app.route('/access_token')
def get_access_token():
    consumer_key = os.getenv('consumer_key')
    consumer_secret = os.getenv('consumer_secret')
    endpoint = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'

    r = requests.get(endpoint, auth=HTTPBasicAuth(consumer_key, consumer_secret))
    data = r.json()
    return data['access_token']






##ADMIN 
####################################
def login_required(f):
	@wraps(f)
	def wrapped(*args, **kwargs):
		if 'authorised' not in session:
			return render_template('admin/login.html')
		return f(*args, **kwargs)
	return wrapped


@app.context_processor
def inject_tables_and_counts():
	data = count_all(mysql)
	return dict(tables_and_counts=data)


@app.route('/admin')
@app.route('/admin')
@login_required
def index():
	return render_template('admin/index.html')


@app.route("/farmers")
@login_required
def farmers():
	data = fetch_all(mysql, "farmers")
	return render_template('admin/farmers.html', data=data, table_count=len(data))


@app.route('/edit_farmers/<string:act>/<int:modifier_id>', methods=['GET', 'POST'])
@login_required
def edit_farmers(modifier_id, act):
	if act == "add":
		return render_template('admin/edit_farmers.html', data="", act="add")
	else:
		data = fetch_one(mysql, "farmers", "id", modifier_id)
	
		if data:
			return render_template('admin/edit_farmers.html', data=data, act=act)
		else:
			return 'Error loading #%s' % modifier_id


@app.route("/admintable")
@login_required
def admintable():
	data = fetch_all(mysql, "admintable")
	return render_template('admin/admintable.html', data=data, table_count=len(data))


@app.route('/edit_admintable/<string:act>/<int:modifier_id>', methods=['GET', 'POST'])
@login_required
def edit_admintable(modifier_id, act):
	if act == "add":
		return render_template('admin/edit_admintable.html', data="", act="add")
	else:
		data = fetch_one(mysql, "admintable", "id", modifier_id)
	
		if data:
			return render_template('admin/edit_admintable.html', data=data, act=act)
		else:
			return 'Error loading #%s' % modifier_id


@app.route('/save', methods=['GET', 'POST'])
@login_required
def save():
	cat = ''
	if request.method == 'POST':
		post_data = request.form.to_dict()
		if 'password' in post_data:
			post_data['password'] = generate_password_hash(post_data['password']) 
		if post_data['act'] == 'add':
			cat = post_data['cat']
			insert_one(mysql, cat, post_data)
		elif post_data['act'] == 'edit':
			cat = post_data['cat']
			update_one(mysql, cat, post_data, post_data['modifier'], post_data['id'])
	else:
		if request.args['act'] == 'delete':
			cat = request.args['cat']
			delete_one(mysql, cat, request.args['modifier'], request.args['id'])
	return redirect("./" + cat)


@app.route('/adminlogin')
def adminlogin():
	if 'authorised' in session:
		return redirect(url_for('admin'))
	else:
		error = request.args['error'] if 'error' in request.args else ''
		return render_template('admin/login.html', error=error)


@app.route('/login_handler', methods=['POST'])
def login_handler():
    email = request.form['email']
    password = request.form['password']
    print(f"Email: {email}, Password: {password}")  # Debug print
    try:
        data = fetch_one(mysql, "admintable", "email", email)
        print(f"Data fetched from database: {data}")  # Debug print
    except Exception as e:
        return render_template('admin/login.html', error=str(e))

    if data and len(data) > 0:
        password_check = check_password_hash(data['password'], password)
        print(f"Password check result: {password_check}")  # Debug print
        if password_check:
            session['authorised'] = 'authorised',
            session['id'] = data['id']
            session['name'] = data['name']
            session['email'] = data['email']
            session['role'] = data['role']
            return redirect(url_for('index'))
        else:
            return redirect(url_for('adminlogin', error='Wrong Email address or Password.'))
    else:
        return redirect(url_for('adminlogin', error='No user'))


@app.route('/adminlogout')
@login_required
def adminlogout():
	session.clear()
	return redirect(url_for('adminlogin'))


def fetch_all(mysql, table_name):
	cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM " + table_name)
	data = cursor.fetchall()
	if data is None:
		return "Problem!"
	else:
		return data


def fetch_one(mysql, table_name, column, value):
	cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM " + table_name + " WHERE " + column + " = '" + str(value) + "'")
	data = cursor.fetchone()
	if data is None:
		return "Problem!"
	else:
		return data


def count_all(mysql):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    data = ()
    
    for table in tables:
        # Check if the table list is not empty
        if table:
            table_name = table['Tables_in_' + app.config['MYSQL_DB']]
            data += ((table_name, count_table(mysql, table_name)),)
    
    return data


def count_table(mysql, table_name):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT COUNT(*) as count FROM " + table_name)
    table_count = cursor.fetchone()
    return table_count['count']


def clean_data(data):
	del data["cat"]
	del data["act"]
	del data["id"]
	del data["modifier"]
	return data


def insert_one(mysql, table_name, data):
    data = clean_data(data)
    columns = ','.join(data.keys())
    values = ','.join([str("'" + e + "'") for e in data.values()])
    insert_command = "INSERT into " + table_name + " (%s) VALUES (%s) " % (columns, values)
    try:
        cursor = mysql.connection.cursor()
        cursor.execute(insert_command)
        mysql.connection.commit()
        return True
    except Exception as e:
        print("Problem inserting into db: " + str(e))
        return False


def update_one(mysql, table_name, data, modifier, item_id):
	data = clean_data(data)
	update_command = "UPDATE " + table_name + " SET {} WHERE " + modifier + " = " + item_id + " LIMIT 1"
	update_command = update_command.format(", ".join("{}= '{}'".format(k, v) for k, v in data.items()))
	try:
		cursor = mysql.connection.cursor()
		cursor.execute(update_command)
		mysql.connection.commit()
		return True
	except Exception as e:
		print("Problem updating into db: " + str(e))
		return False


def delete_one(mysql, table_name, modifier, item_id):
	try:
		cursor = mysql.connection.cursor()
		delete_command = "DELETE FROM " + table_name + " WHERE " + modifier + " = " + item_id + " LIMIT 1"
		cursor.execute(delete_command)
		mysql.connection.commit()
		return True
	except Exception as e:
		print("Problem deleting from db: " + str(e))
		return False


if __name__ == "__main__":
	app.run(debug=True)