A plant disease predictor app using python,CNN and flask

# Project Avodoc
## Background information
A plant disease predictor app using python,CNN and flask


## Tech stack overview
Below is a brief overview of the tech stack leveraged to bring Avodoc to life.
### Frontend

- `Jinja2` templating
- `Bootstrap` CSS styling
- `SASS`(Syntactically Awesome Style Sheets)

### Backend

- `Flask-Python3`
- `Mysql` for development and `MySQL` for production.
- `nginx` webserver for static data and `gunicorn` for serving the dynamic application contents.


# Installation
### Prerequisites
- Ubuntu 20.04 LTS - Operating system required.

This project was developed and tested on an `Ubuntu 20.04 LTS` terminal. Using other Ubuntu versions may result in some
incompatibility issues. If you're not on an Ubuntu 20.04 LTS terminal/os/VM, I'd suggest using a `docker` container spinning the Ubuntu 20.04 LTS image for full functionality of the app.

- Python3 - Installed in your local terminal/vagrant/VM/docker container

### Getting started
Clone the repository to your local terminal, Ubuntu 20.04 LTS remember, then create a virtual environment using:
`Python3 -m venv venv` then launch that virtual environment while you're in the repo's root directory with this command:
`source venv/bin/activate`. You'll need this virtual environment to run the application successfully with all it's required packages without affecting any of your previously globally installed packages in your local machine.
#### NOTE:
You will have to configure the environment variables with your own values in order to run the application. 

Once you're in the virtual environment, you can install the rest of the packages required to run the application located in the `requirements.txt` file. Use this command:
`pip install -r requirements.txt` 


# Usage

Now you're ready to start running the application locally(in the development server) in your machine.
You can run it using either of these two commands:
  - `python app.py` or
  - `flask run`
It'll be listening on port 5000 by default. You can browse it in your browser and test out it's awesome features and functionalities.


# Contribution

All contributions to help improve the application's features and functionalities are welcome. Fork the repository and create a pull request with your modifications. I'll be sure to review them.


# Authors

- Prince Tobit - [Github](https://github.com/tobitprince) / [LinkedIn](https://www.linkedin.com/in/prince-tobit-820060259/) / [X](https://twitter.com/tobitprince)  


# License🧾📜

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.