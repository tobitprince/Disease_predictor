from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re


app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e6d7g8h9i10'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234' #Replace ******* with  your database password.
app.config['MYSQL_DB'] = 'pythonlogin'


# Intialize MySQL
mysql = MySQL(app)


@ app.route('/')
def home():
    print("mysql.connection:", mysql.connection)  # Print the value of mysql.connection
    return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)