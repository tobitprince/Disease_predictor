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
from werkzeug.security import generate_password_hash
import base64
from flask_mail import Mail, Message
import jwt
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itsdangerous import URLSafeTimedSerializer
from werkzeug.security import check_password_hash


app = Flask(__name__)
load_dotenv()

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = os.getenv('app.secret_key')

# access token
consumerKey = os.getenv('consumerKey') #Fill with your app Consumer Key
consumerSecret = os.getenv('consumerSecret') # Fill with your app Secret
base_url = os.getenv('base_url')

##Set up the configuration for flask_mail.
app.config['MAIL_SERVER']=os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
##update it with your gmail
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
##update it with your password
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
##app.config["EMAIL_SENDER"] = os.getenv('MAIL_SENDER')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS')
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL')


# Enter your database connection details below
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')


# Intialize MySQL
mysql = MySQL(app)

#Create an instance of Mail.
mail = Mail(app)

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
    return render_template('crop.html',username=session['username'])

@ app.route('/login')
def login():
    return render_template('login/signup-login.html')

# http://localhost:5000/login/ - this will be the login page, we need to use both GET and POST requests
@app.route('/logini', methods=['GET', 'POST'])
def logini():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']


        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute('SELECT * FROM farmers WHERE email = %s', (email,))
        account = cursor.fetchone()
        print(account['password'])

        if account and check_password_hash(account['password'], password):
            cursor.execute('SELECT * FROM farmers WHERE email = %s AND status = 1', (email,))
            verified_account = cursor.fetchone()
            if verified_account:
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']

                mysql.connection.commit()

                return redirect(url_for('home2'))
            else:
                msg = 'Account not verified!'
                flash("Account not verified!", "danger")
        else:
            msg = 'Incorrect username/password!'
            flash("Incorrect username/password!", "danger")

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
        email_address = request.form['email']
        status = 0
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM farmers WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM farmers WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email_address):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email_address:
            flash("Incorrect username/password!", "danger")
        else:
    
            # Hash the password
            password = generate_password_hash(request.form['password'])
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO farmers VALUES (NULL, %s, %s, %s, %s)', (username,email_address, password, status))
            mysql.connection.commit()
            token = jwt.encode(
                {
                    "email_address": email_address,
                    "password": password,
                }, os.getenv('app.secret_key')
            )

            # Ensure the token is a byte string
            if isinstance(token, str):
                token = token.encode()

            # Now you can decode the token
            decoded_data = jwt.decode(token, os.getenv('app.secret_key'), algorithms=["HS256"])
            print(f"Type of email_address: {type(email_address)}")
            port = os.getenv('port')  # For starttls
            smtp_server = os.getenv('smtp_server')
            sender_email = os.getenv('sender_email')
            receiver_email = email_address
            password = os.getenv('password')
            try:
                # Convert the token to a string
                token_str = token.decode('utf-8')
                message = MIMEMultipart("alternative")
                message["Subject"] = "OTP"
                message["From"] = sender_email
                message["To"] = receiver_email

                # Create the plain-text and HTML version of your message
                text = """\
                Hi,
                Dear user, Your verification OTP code is {token}
                With regards,
                Avodoc""".format(token=token_str)
                html = """\
                <html>
                <body>
                    <p>Hi,<br>
                    Dear user, </p> <h3>Your verification OTP code is </h3>
                    <br><br>
                      {token}
                    </p>
                    <br><br>
                    <p>With regards,</p>
                    <b>Avodoc</b>
                </body>
                </html>
                """.format(token=token_str)

                # Turn these into plain/html MIMEText objects
                part1 = MIMEText(text, "plain")
                part2 = MIMEText(html, "html")

                # Add HTML/plain-text parts to MIMEMultipart message
                # The email client will try to render the last part first
                message.attach(part1)
                message.attach(part2)

                context = ssl.create_default_context()
                with smtplib.SMTP(smtp_server, port) as server:
                    server.ehlo()  # Can be omitted
                    server.starttls(context=context)
                    server.ehlo()  # Can be omitted
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.as_string())
                
            except Exception as e:
                print(f"Error sending email: {e}")
            return render_template("login/verify_email.html")

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Fill form'
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    msg = 'error'
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

    return render_template('disease.html',username=session['username'])

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

############OTP
@app.route("/verify-email", methods=['GET', 'POST'])
def verify_email():
    msg = ''
    try:
        if request.method == 'POST':
            token = request.form['token']
            data = jwt.decode(token, os.getenv('app.secret_key'),algorithms=["HS256"])
            email_address = data["email_address"]
            password = data["password"]
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("UPDATE farmers SET status = 1 WHERE email = %s", (email_address,))
            mysql.connection.commit()

            msg = 'Account verified'
            flash("Your account has successfully been registered!", "success")
            return render_template('login/signup-login.html', msg=msg)
        elif request.method == 'GET':
            return render_template('login/verify_email.html', msg=msg)
    except jwt.DecodeError:
        flash("Invalid token!", "danger")
        return render_template('login/verify_email.html', msg='Invalid token')
    except Exception as e:
        flash(str(e), "danger")
        return render_template('login/verify_email.html', msg='An error occurred')


###################
####Recover########
###################
@app.route('/forgot_password')
def forgot_password():
     return render_template("login/recover.html")
@app.route('/recover', methods=['GET', 'POST'])
def recover():
    if request.method == 'POST':
        email_address = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM farmers WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM farmers WHERE email LIKE %s", [email_address] )
        account = cursor.fetchone()

        if account:
            serializer = URLSafeTimedSerializer(os.getenv('app.secret_key'))
            token = serializer.dumps(email_address, salt = os.getenv('salt'))

            port = os.getenv('port')  # For starttls
            smtp_server = os.getenv('smtp_server')
            sender_email = os.getenv('sender_email')
            receiver_email = email_address
            password = os.getenv('password')

            # Convert the token to a string
            #token_str = token.decode('utf-8')

            link = url_for('reset_with_token', token=token, _external=True)


            try:
                
                message = MIMEMultipart("alternative")
                message["Subject"] = "Password Reset Request"
                message["From"] = sender_email
                message["To"] = receiver_email

                # Create the plain-text and HTML version of your message
                text = """\
                Hi,
                Your link is {}
                With regards,
                Avodoc""".format(link)
                html = """\
                <html>
                <body>
                    <p>Hi,<br>
                    Dear user, </p> <h3>Your link is </h3>
                    <br><br>
                      {}
                    </p>
                    <br><br>
                    <p>With regards,</p>
                    <b>Avodoc</b>
                </body>
                </html>
                """.format(link)

                # Turn these into plain/html MIMEText objects
                part1 = MIMEText(text, "plain")
                part2 = MIMEText(html, "html")

                # Add HTML/plain-text parts to MIMEMultipart message
                # The email client will try to render the last part first
                message.attach(part1)
                message.attach(part2)

                context = ssl.create_default_context()
                with smtplib.SMTP(smtp_server, port) as server:
                    server.ehlo()  # Can be omitted
                    server.starttls(context=context)
                    server.ehlo()  # Can be omitted
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.as_string())
                flash('An email has been sent with instructions to reset your password.', 'success')
                
            except Exception as e:
                print(f"Error sending email: {e}")
                return render_template("login/recover.html")
        else:
            flash('No account found for that email address.', 'danger')

    return render_template('login/recover.html')

@app.route('/reset/<token>', methods=['GET', 'POST'])
def reset_with_token(token):
    try:
        serializer = URLSafeTimedSerializer(os.getenv('app.secret_key'))
        email_address = serializer.loads(token, salt=os.getenv('salt'), max_age=3600)
        return render_template('login/reset_with_token.html', token=token, _external=True)
    except:
        flash('The password reset link is invalid or has expired.', 'danger')
        return redirect(url_for('recover'))
    
@app.route('/rst', methods = ['GET', 'POST'])
def rst():
    if request.method == 'POST':
        token = request.form['token']
        serializer = URLSafeTimedSerializer(os.getenv('app.secret_key'))
        email_address = serializer.loads(token, salt=os.getenv('salt'), max_age=3600)
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('login/reset_with_token.html')
        # Hash the password
        password = generate_password_hash(request.form['password'])
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("UPDATE farmers SET password = %s WHERE email = %s", (password, email_address,))
        mysql.connection.commit()

        flash('Your password has been updated!', 'success')
        return redirect(url_for('login'))

    return render_template('login/reset_with_token.html')
                  




#####MPESAAA

@ app.route('/mpesa')
def mpesa():
    return render_template('mpesa/index.html')

#intitate mpesa request
@app.route('/pay', methods=['GET', 'POST'])
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
            password = base64.b64encode(password.encode('utf-8')).decode('utf-8')

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
      else:
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
    if r.status_code == 200:
        try:
            data = r.json()
            return data['access_token']
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            print(f"Response: {r.text}")
            return "Error occurred"
    else:
        print(f"Error: Received status code {r.status_code}")
        print(r.text)  # print the response text
        return "Error occurred"


###################
####CONTACT########
###################
@app.route('/contact', methods=['GET', 'POST'])
def contact():
     if request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'subject' in request.form and 'message' in request.form:
            name = request.form['name']
            email_address = request.form['email']
            subject = request.form['subject']
            message = request.form['message']

            port = os.getenv('port')  # For starttls
            smtp_server = os.getenv('smtp_server')
            sender_email = os.getenv('sender_email')
            receiver_email = email_address
            password = os.getenv('password')
            try:
                message = MIMEMultipart("alternative")
                message["Subject"] = subject
                message["From"] = sender_email
                message["To"] = receiver_email

                # Create the plain-text and HTML version of your message
                text = """\
                Hi {name},
                Thank you for contacting us and we will attend to you right away
                With regards,
                Avodoc""".format(name = name)
                html = """\
                <html>
                <body>
                    <p>Hi {name},<br>
                    Thank you for contacting us and we will attend to you right away
                    <br><br>
                    </p>
                    <br><br>
                    <p>With regards,</p>
                    <b>Avodoc</b>
                </body>
                </html>
                """.format(name = name)

                # Turn these into plain/html MIMEText objects
                part1 = MIMEText(text, "plain")
                part2 = MIMEText(html, "html")

                # Add HTML/plain-text parts to MIMEMultipart message
                # The email client will try to render the last part first
                message.attach(part1)
                message.attach(part2)

                context = ssl.create_default_context()
                with smtplib.SMTP(smtp_server, port) as server:
                    server.ehlo()  # Can be omitted
                    server.starttls(context=context)
                    server.ehlo()  # Can be omitted
                    server.login(sender_email, password)
                    server.sendmail(sender_email, receiver_email, message.as_string())
                flash('Your message has been sent', 'success')
            except Exception as e:
                print(f"Error sending email: {e}")
                flash('Error sending email', 'danger')
            return redirect(url_for('home2'))
     return render_template('index3.html')





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