# Python 中的开发环境

> 原文：<https://www.pythonforbeginners.com/development/development-environment-in-python>

## 概观

```py
 Some of the steps needed to setup a development environment includes:

Operating system 	- e.g Linux / Mac
Project structure  	- project structure
Virtualenv   		- isolated installation of the project
Pip     		- a tool for installing and managing Python packages
Git     		- source control
Webserver     		- where we can manage our applications
Fabric    		- automated deployment 
```

## 项目结构

```py
Create an empty top-level directory for our new project. 
```

```py
 helloflask/
    -static/
        -css
        -font
        -img
        -js
    -templates
    -routes.py

**Then cd into the directory**
cd helloflask 
```

## Virtualenv(虚拟环境)

```py
 Many developers uses virtualenv (virtual environment) on their computer, which
is useful when you want to run several applications on the same computer. 

Virtualenv will manage all dependencies and enables multiple side-by-side
installations of Python, one for each project.

It doesn't install separate copies of Python, but provides a way to keep
different project environments isolated.

If we want to run more than one (which is often the case) web application on
that host, then you should really install 'Virtualenv'. 

If you don't use virtualenv , you will have it all globally installed. 
```

#### 安装 Virtualenv

```py
 Download and Install Virtualenv into a virtual environment 
```

```py
# If you are using Linux/Mac:
sudo pip install virtualenv

```

#### 设置新项目

```py
 Navigate to the directory you want your project in: 
```

```py
$ virtualenv venv      		# this creates the folder venv
$ source venv/bin/activate    	# start working on your new project
(venv)$ pip install Flask    	# installs Flask

```

```py
 For more information on how to download install virtualenv, see [this](https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv "virtualenv_install") article.
```

## 点

```py
 PIP is a tool for installing and managing Python packages.

PIP comes with a command-line interface, which makes installing Python software
packages as easy as issuing one command 
```

```py
pip install some-package-name

```

```py
 Users can also easily implement the package's subsequent removal 
```

```py
pip uninstall some-package-name

```

```py
 Pip has a feature to manage full lists of packages and corresponding version
numbers through a "requirements" file.

This permits the efficient re-creation of an entire group of packages in a
separate environment (e.g. another computer) or virtual environment. 

This can be achieved with a properly formatted requirements.txt file 
```

```py
pip install -r requirements.txt

```

```py
 This makes dependencies easy, you can create a requirements file based on a set of
packages installed in your virtual environment. 
```

```py
pip freeze > requirements.txt

```

```py
 When deploying to a server it is important to register which requirements we need. 

The requirements file can be done automatically using the freeze command for pip. 

This command will generate a plain text file that contains the names of the
required Python packages and their versions, for example **Flask==0.9** 
```

```py
 To do this we freeze the installed packages and store this setup in a
requirements.txt file 
```

```py
$ cat requirements.txt
Flask==0.9
Jinja2==2.6
Werkzeug==0.8.3

```

```py
 The requirements file can be used to rebuild a virtual environment or to deploy a
virtual environment into the machine. 
```

## 开始编码

```py
 Now that we have a clean Flask environment to work in, we'll create our simple
application. 

The simplest Flask App looks something like this: 
```

```py
 Put this code into the file and name it 'hello.py' 
```

```py
from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello World!'
```

## github–中央存储库

```py
 Now it's time to create the repository on Github. The purpose of setting up a
Github project, is so that we can push files from our local computer to Github
and then pull the files from Github to our web server.

Create a new Github account and create a new project (helloflask) 
```

## git–本地计算机

```py
 By using a versioning system, we can store all our files in a Github repository.

The first thing you need to do on your **local computer** is to install and setup git. 
```

##### Install Git

```py
 To install git, simple run: 
```

```py
sudo apt-get install git

```

##### 进入设置

```py
 Put in your username and email into the .gitconfig file (~/.gitconfig) 
```

```py
git config --global user.name "pythonforbeginners"
git config --global user.email [email protected]

```

##### Git 忽略

```py
 Since our current directory contains a lof of extra files, we'll want to
configure our repository to ignore these files with a **.gitignore** file:
venv
*.pyc 
```

```py
 Next, we’ll create a new git repository and save our changes. 
```

```py
# Initialize Git in our project directory
git init

```

```py
 This creates a git repository in the current directory. 
```

```py
 Add all of our files to our initial commit 
```

```py
git add .

```

```py
 Check the status, this will list all files 
```

```py
git status

```

```py
 With the files added to the Git index, we can now **commit** them to our repo: 
```

```py
$ git commit -m 'Initial commit'

```

```py
 Now we have created a local Git repository for our application (local) files. 
```

```py
Setup Github as the origin 
```

```py
git remote add origin [email protected]:USERNAME/helloflask.git
git push -u origin master

```

## Web 服务器–主机

```py
 Now its time to start up the web server and do some configuration. 

If you want to use **Apache** as a web server, you can install it like this: 
```

```py
sudo apt-get install apache2

```

```py
 Configure a **virtual host** (vhost) in /etc/apache2/sites-available/siteX

Install **virtualenv** just like you did on your local computer. 

Set up the environment for the website, here I use /var/www

Cd into that folder and **clone the project** that you setup on Github, by typing: 
```

```py
git clone [email protected]:USERNAME/helloflask.git

```

```py
 Initialize and activate your virtualenv 
```

```py
virtualenv helloflask
cd helloflask
source bin/activate

```

```py
Install dependencies
```

```py
pip install -r requirements.txt

```

## 构造

```py
 Fabric is used for deployment. You can of course always manually upload the code
and restart the web server to reflect the configuration changes.

Using fabric in a development environment, where you have multiple servers with
multiple people pushing the code multiple times per day, this can be incredible
very useful.

Fabric can configure the system, execute commands on local/remote server, deploy
your application, do rollbacks etc. 

It does that by using its command-line utility that will run a fabfile containing
instructions on how to deploy to a server.

A common practice when developing is to use Git to deploy and Fabric to automate
it. 
```

#### 安装结构

```py
pip install fabric

```

```py
 Fabric expects a fabfile named fabfile.py which defines all of the actions we can
take. 

The fabfile.py should be in your project's root directory.

I like to use this script that asks the server to pull from the git repository
and restart apache. [[source]](https://yuji.wordpress.com/2011/04/09/django-python-fabric-deployment-script-and-example/ "fabric_script") 
```

```py
from fabric.api import *            # import fabrics API functions
env.hosts = ['[email protected]:22'] # add the remote server information 
def pushpull():
    local('git push')      		    # runs the command on the local environment
    run('cd /path/to/project/; git pull') # runs the command on the remote environment 

```

```py
#Run it with:
$ fab pushpull

```

```py
For more information on how to use fabric in a development environment, please
refer to [this](https://www.pythonforbeginners.com/systems-programming/how-to-use-fabric-in-a-development-environment "fabric_in_a_dev") article. 
```