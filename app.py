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

#from android.permissions import Permission, request_permission
#request_permission([Permission.READ_EXTERNAL_STOARGE,Permission.WRITE_EXTERNAL_STOARGE])


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
    return render_template('login/signup-login.html')

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

@ app.route('/mpesa')
def mpesa():
    return render_template('mpesa/index.html')
   

@ app.route('/mpesastk')
def mpesastk():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:


    datez = date_default_timezone_set('Africa/Nairobi')

    

    # define the variales
    # provide the following details, this part is found on your test credentials on the developer account
    BusinessShortCode = '174379'
    Passkey = 'bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919';  
  
    
    #This are your info, for
    #PartyA should be the ACTUAL clients phone number or your phone number, format 2547********
    #AccountRefference, it maybe invoice number, account number etc on production systems, but for test just put anything
    #TransactionDesc can be anything, probably a better description of or the transaction
    #Amount this is the total invoiced amount, Any amount here will be 
    #actually deducted from a clients side/your test phone number once the PIN has been entered to authorize the transaction. 
    #for developer/test accounts, this money will be reversed automatically by midnight.
   
    PartyA = request.form['phone'] # This is your phone number, 
    AccountReference = '2255'
    TransactionDesc = 'Test Payment'
    Amount = request.form['amount']
 
    # Get the timestamp, format YYYYmmddhms -> 20181004151020
    Timestamp = date('YmdHis');    
  
    # Get the base64 encoded string -> $password. The passkey is the M-PESA Public Key
    Password = base64_encode(BusinessShortCode.Passkey.Timestamp)

    # header for access token
    headers = ['Content-Type:application/json; charset=utf8'];

    # M-PESA endpoint urls
    access_token_url = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials';
    initiate_url = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest';

    # callback url
    CallBackURL = 'https://avodoc-6ab21772621e.herokuapp.com/mpesa/index.php';  

    curl = curl_init(access_token_url)
    curl_setopt(curl, CURLOPT_HTTPHEADER, headers)
    curl_setopt(curl, CURLOPT_RETURNTRANSFER, TRUE)
    curl_setopt(curl, CURLOPT_HEADER, FALSE)
    curl_setopt(curl, CURLOPT_USERPWD, consumerKey.':'.consumerSecret)
    result = curl_exec(curl)
    status = curl_getinfo(curl, CURLINFO_HTTP_CODE)
    result = json_decode(result)
    access_token = result->access_token;  
    curl_close(curl)

    # header for stk push
    stkheader = ['Content-Type:application/json','Authorization:Bearer '.access_token]

    # initiating the transaction
    curl = curl_init()
    curl_setopt(curl, CURLOPT_URL, initiate_url)
    curl_setopt(curl, CURLOPT_HTTPHEADER, stkheader); #setting custom header

    curl_post_data = array(
        #Fill in the request parameters with valid values
        'BusinessShortCode' => BusinessShortCode,
        'Password' => Password,
        'Timestamp' => Timestamp,
        'TransactionType' => 'CustomerPayBillOnline',
        'Amount' => Amount,
        'PartyA' => PartyA,
        'PartyB' => BusinessShortCode,
        'PhoneNumber' => PartyA,
        'CallBackURL' => CallBackURL,
        'AccountReference' => AccountReference,
        'TransactionDesc' => TransactionDesc
    )

    data_string = json_encode(curl_post_data)
    curl_setopt(curl, CURLOPT_RETURNTRANSFER, true)
    curl_setopt(curl, CURLOPT_POST, true)
    curl_setopt(curl, CURLOPT_POSTFIELDS, data_string)
    curl_response = curl_exec(curl)
    print_r(curl_response)

    echo curl_response
        return render_template('mpesa/index.html')
######################################################################
@app.route('/access_token')
def get_access_token():
    consumer_key = consumer_key
    consumer_secret = consumer_secret
    endpoint = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'

    r = requests.get(endpoint, auth=HTTPBasicAuth(consumer_key, consumer_secret))
    data = r.json()
    return data['access_token']

@app.route('/register')
def register_urls():
    endpoint = 'https://sandbox.safaricom.co.ke/mpesa/c2b/v1/registerurl'
    access_token = _access_token()
    my_endpoint = base_url + "c2b/"
    headers = { "Authorization": "Bearer %s" % access_token }
    r_data = {
        "ShortCode": "174379",
        "ResponseType": "Completed",
        "ConfirmationURL": my_endpoint + 'con',
        "ValidationURL": my_endpoint + 'val'
    }

    response = requests.post(endpoint, json = r_data, headers = headers)
    return response.json()


@app.route('/simulate')
def test_payment():
    endpoint = 'https://sandbox.safaricom.co.ke/mpesa/c2b/v1/simulate'
    access_token = _access_token()
    headers = { "Authorization": "Bearer %s" % access_token }

    data_s = {
        "Amount": 100,
        "ShortCode": "174379",
        "BillRefNumber": "test",
        "CommandID": "CustomerPayBillOnline",
        "Msisdn": "254708374149"
    }

    res = requests.post(endpoint, json= data_s, headers = headers)
    return res.json()

@app.route('/b2c')
def make_payment():
    endpoint = 'https://sandbox.safaricom.co.ke/mpesa/b2c/v1/paymentrequest'
    access_token = _access_token()
    headers = { "Authorization": "Bearer %s" % access_token }
    my_endpoint = base_url + "/b2c/"

    data = {
        "InitiatorName": "apitest342",
        "SecurityCredential": "SQFrXJpsdlADCsa986yt5KIVhkskagK+1UGBnfSu4Gp26eFRLM2eyNZeNvsqQhY9yHfNECES3xyxOWK/mG57Xsiw9skCI9egn5RvrzHOaijfe3VxVjA7S0+YYluzFpF6OO7Cw9qxiIlynYS0zI3NWv2F8HxJHj81y2Ix9WodKmCw68BT8KDge4OUMVo3BDN2XVv794T6J82t3/hPwkIRyJ1o5wC2teSQTgob1lDBXI5AwgbifDKe/7Y3p2nn7KCebNmRVwnsVwtcjgFs78+2wDtHF2HVwZBedmbnm7j09JO9cK8glTikiz6H7v0vcQO19HcyDw62psJcV2c4HDncWw==",
        "CommandID": "BusinessPayment",
        "Amount": "200",
        "PartyA": "174379",
        "PartyB": "254708374149",
        "Remarks": "Pay Salary",
        "QueueTimeOutURL": my_endpoint + "timeout",
        "ResultURL": my_endpoint + "result",
        "Occasion": "Salary"
    }

    res = requests.post(endpoint, json = data, headers = headers)
    return res.json()

@app.route('/lnmo')
def init_stk():
    endpoint = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest'
    access_token = _access_token()
    headers = { "Authorization": "Bearer %s" % access_token }
    my_endpoint = base_url + "/lnmo"
    Timestamp = datetime.now()
    times = Timestamp.strftime("%Y%m%d%H%M%S")
    password = "174379" + "bfb279f9aa9bdbcf158e97dd71a467cd2e0c893059b10f78e6b72ada1ed2c919" + times
    datapass = base64.b64encode(password.encode('utf-8'))

    data = {
        "BusinessShortCode": "174379",
        "Password": datapass,
        "Timestamp": times,
        "TransactionType": "CustomerPayBillOnline",
        "PartyA": request.form['phone'], # fill with your phone number
        "PartyB": "174379",
        "PhoneNumber": request.form['phone'], # fill with your phone number
        "CallBackURL": my_endpoint,
        "AccountReference": "TestPay",
        "TransactionDesc": "HelloTest",
        "Amount": 2
    }

    res = requests.post(endpoint, json = data, headers = headers)
    return res.json()

@app.route('/lnmo', methods=['POST'])
def lnmo_result():
    data = request.get_data()
    f = open('lnmo.json', 'a')
    f.write(data)
    f.close()

@app.route('/b2c/result', methods=['POST'])
def result_b2c():
    data = request.get_data()
    f = open('b2c.json', 'a')
    f.write(data)
    f.close()

@app.route('/b2c/timeout', methods=['POST'])
def b2c_timeout():
    data = request.get_json()
    f = open('b2ctimeout.json', 'a')
    f.write(data)
    f.close()

@app.route('/c2b/val', methods=['POST'])
def validate():
    data = request.get_data()
    f = open('data_v.json', 'a')
    f.write(data)
    f.close()

@app.route('/c2b/con', methods=['POST'])
def confirm():
    data = request.get_json()
    f = open('data_c.json', 'a')
    f.write(data)
    f.close()


def _access_token():
    consumer_key = consumer_key
    consumer_secret = consumer_secret
    endpoint = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'

    r = requests.get(endpoint, auth=HTTPBasicAuth(consumer_key, consumer_secret))
    data = r.json()
    return data['access_token']







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


@app.route("/adminold")
@login_required
def adminold():
	data = fetch_all(mysql, "adminold")
	return render_template('admin/adminold.html', data=data, table_count=len(data))


@app.route('/edit_adminold/<string:act>/<int:modifier_id>', methods=['GET', 'POST'])
@login_required
def edit_adminold(modifier_id, act):
	if act == "add":
		return render_template('admin/edit_adminold.html', data="", act="add")
	else:
		data = fetch_one(mysql, "adminold", "id", modifier_id)
	
		if data:
			return render_template('admin/edit_adminold.html', data=data, act=act)
		else:
			return 'Error loading #%s' % modifier_id


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


@app.route("/users")
@login_required
def users():
	data = fetch_all(mysql, "users")
	return render_template('admin/users.html', data=data, table_count=len(data))


@app.route('/edit_users/<string:act>/<int:modifier_id>', methods=['GET', 'POST'])
@login_required
def edit_users(modifier_id, act):
	if act == "add":
		return render_template('admin/edit_users.html', data="", act="add")
	else:
		data = fetch_one(mysql, "users", "id", modifier_id)
	
		if data:
			return render_template('admin/edit_users.html', data=data, act=act)
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
def login():
	if 'authorised' in session:
		return redirect(url_for('admin'))
	else:
		error = request.args['error'] if 'error' in request.args else ''
		return render_template('admin/login.html', error=error)


@app.route('/login_handler', methods=['POST'])
def login_handler():
	try:
		email = request.form['email']
		password = request.form['password']
		data = fetch_one(mysql, "users", "email", email)
		
		if data and len(data) > 0:
			if check_password_hash(data[3], password) or hashlib.md5(password.encode('utf-8')).hexdigest() == data[3]:
				session['authorised'] = 'authorised',
				session['id'] = data[0]
				session['name'] = data[1]
				session['email'] = data[2]
				session['role'] = data[4]
				return redirect(url_for('index'))
			else:
				return redirect(url_for('login', error='Wrong Email address or Password.'))
		else:
			return redirect(url_for('login', error='No user'))
	
	except Exception as e:
		return render_template('admin/login.html', error=str(e))


@app.route('/adminlogout')
@login_required
def logout():
	session.clear()
	return redirect(url_for('adminlogin'))

if __name__ == "__main__":
	app.run(debug=True)