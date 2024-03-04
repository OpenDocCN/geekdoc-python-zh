# 如何在开发环境中使用 Fabric

> 原文：<https://www.pythonforbeginners.com/systems-programming/how-to-use-fabric-in-a-development-environment>

## 概观

```py
I earlier wrote a post on "How to use Fabric in Python", which can be found [here](../../../../systems-programming/how-to-use-fabric-in-python/ "fabric_usage"). 

I received a lot of responses from that article, so I decided to write another
post about Fabric. This time I would like to focus on how to use it in a
development environment. 
```

## 什么是面料？

```py
**Just to recap what Fabric is**

Fabric is a Python (2.5 or higher) library and command-line tool for streamlining
the use of SSH for application deployment or systems administration tasks.

It provides a basic suite of operations for executing local or remote shell
commands (normally or via sudo) and uploading/downloading files, as well as
auxiliary functionality such as prompting the running user for input, or aborting
execution. 
```

## 为什么在开发环境中使用 Fabric？

```py
So, we can use Fabric for streamlining the use of SSH for application deployment
or systems administration tasks.

In a development environment, where you have multiple servers with multiple people
pushing the code multiple times per day, this can be very useful.

If you would be the only person in the project who worked on a single server,
a git pull *(A git pull is what you would do to bring your repository up to date
with a remote repository)* and start working with it.

Often, you are not the only developer in a projet and there is where Fabric comes
into the picture. To avoid logging into multiple servers and running remote
commands, we can use Fabric to automate this whole process.

It can configure the system, execute commands on local/remote server, deploy your
application, do rollbacks etc. A common practice when developing is to use Git to
deploy and Fabric to automate it. 
```

## 织物要求

```py
An SSH server like OpenSSH needs to be installed on your server and an SSH client
needs to be installed on your client.

Fabric relies on the SSH Model, you can use SSH Keys but you can also control
access to root via sudoers. 

The user on the server does not need to be added to "~/.ssh/authorized_keys",
but if it is you don't have to type the password every time you want to execute
a command.

If you ever have to disable the access to a user, just turn off their SSH account.

There is no transfer of code, so you only need to have ssh running on the remote
machine and have some sort of shell (bash is assumed by default). 
```

## 织物安装

```py
First off, we will have to install it, that can be done through pip 
```

```py
pip install fabric 
```

## Fabric 和 Fabfile.py

```py
The installation added a Python script called **"fab"** to a directory in your path. 

This is the **command** Fabric will use when it logs into one or more servers.

The **fabile.py** (see below) will be executed by the fab command.

All fabric does at its core is execute commands locally and remotely.

Fabric just provides a nice pythonic way of doing it. 
```

```py
The fabile is what Fabric uses to to **exectue task** and we need to specify which
shell commands we want Fabric to execute. In short, a fabfile is what controls
what Fabric executes. 

The commands that we put into the fabfile will be  executed on one or more hosts. 

Every fabric scripts need to be in a file called fabfile.py. These hosts can be
defined either in the fabfile or on the command line.

All the functions defined in this file will show up as fab subcommands. 
```

## 织物功能

```py
Fabric provides a set of commands in **fabric.api** that are simple but powerful,
below are some of them 
```

```py
**local**	# execute a local command)
**run**	# execute a remote command on all specific hosts, user-level permissions)
**sudo**	# sudo a command on the remote server)
**cd**	# changes the directory on the serverside, use with the "with" statement)
**put**  	# uploads a local file to a remote server)
**get** 	# download a file from the remote server)
**prompt**	# prompt user with text and return the input (like raw_input))
**reboot**	# reboot the remote system, disconnect, and wait for wait seconds) 
```

```py
One of the most common Fabric functions is **run()**

It executes whatever command you put in as a parameter on the server. 

Below you can see an example. 
```

```py
run("uptime") 
```

```py
This will run the uptime command on any server that we **specify** in the fabfile.py.

The server(s) are set using **env.hosts:** 
```

```py
env.hosts = ['[[email protected]](/cdn-cgi/l/email-protection)'] 
```

```py
This tells the script to use the username **'superuser**' with the server address
**'host1.example.com'**

Basically, it's just like **ssh'ing** into a box and running the **commands** you've put
into run() and sudo(), but Fabric allows you to use several more **calls**. 
```

## 设置开发环境

```py
Before we start with creating Fabric functions, I want to show an example of how
a **directory structure** may look like: 
```

```py
$ mkdir /var/www/yourapplication
$ cd /var/www/yourapplication
$ virtualenv --distribute env 
```

```py
Before we can do anything, we first need to create a **fabile.py**.

If you are located in the same directory as "fabfile.py" you can go **"fab --list"**
to see a list of available commands and then **"fab [COMMAND_NAME]"** to execute a
command.

Now when everything is setup and we have an fabfile.py, let's put something in it. 
```

```py
Let's first start **defining roles** and then specify what action to do. 
```

```py
from fabric.api import *

# Define sets of servers as roles
env.roledefs = {
    'www': ['www1', 'www2', 'www3', 'www4', 'www5'],
    'databases': ['db1', 'db2']
}

# Set the user to use for ssh
env.user = 'fabuser'

# Restrict the function to the 'www' role (that we created above)
@roles('www')
def uptime():
    run('uptime')

If we now want to see the uptime on all our web-servers, all we have to do is run: 
$ fab -R www

# To run uptime on both roles (www and databases);
$ fab uptime 
```

```py
Any function you write in your fab script can be assigned to one or more roles. 

You can also include a single server in multiple roles.

So this is nice, Fabric makes it really easy to run commands across sets of
machines. 
```

## 使用结构进行部署

```py
Let's see now how we can deploy something with Fabric. Often, we want to create
more than one function, depending on how your environment looks like (live,
staging, dev) 
```

```py
 env.roledefs = {
    'hostsDev': ['dev1', 'dev2'],
    'hostsStaging':[ 'stg1', 'stg2'],
}

@roles('hostsDev')

def deploy_dev():

  # Specify the path to where your codebase is located
  path = "/home/fabuser/codebase"

  # Get the code from Github
  run ("git clone https://www.github.com/user/repo.git")

@roles('hostsStaging')

def deploy_staging():
    # do something 
```

```py
If you now want to do a git clone to your development servers, all you need to do
is use the fab command:
**$ fab deploy_dev**

That's how you can create a deploy function using **Fabric**. 
```

## 虚拟环境和要求

```py
Now we have cloned our code from the Github repository. That is nice, but we can
do more, we can also install the requirements into a virtual environment. 

For those of you that don't know what "requirements" is; 
It's basically a file called **requirements.txt** which contain a list of packages
to install. 
```

```py
Instead of running something like pip install MyApp and getting whatever libraries
come along, you can create a requirements file that includes all dependencies.

Example of a requirements.txt file 
```

```py
BeautifulSoup==3.2.0
python-dateutil==1.4.1
django==1.3.0
django-debug-toolbar==0.8.5
django-tagging==0.3
Markdown==2.0.1
...
... 
```

```py
To install all packages that are in that file, simple type 
```

```py
pip install -r requirements.txt 
```

```py
A **virtual environment** can be created with **/bin/virtualenv 'app_name'** and
activated by typing **source /bin/activate** 
```

## 使用 Fabric 从 requirements.txt 安装软件包

```py
Having that information, let's create a Fabric script to install it. 
```

```py
# Import the module
from fabric.api import *

# Specify the user and host
env.hosts = ['[[email protected]](/cdn-cgi/l/email-protection)']

def deploy():
     with cd("/home/fabuser/codebase/"):

          run("git clone https://github.com/user/repo.git")

     run("source /home/fabuser/codebase/venv/bin/activate && pip install -r /path/to/requirements.txt") 
```

```py
That small script will activate the virtual environment and install all packages
specified in the requirements.txt file. 

Sample output 
```

```py
Downloading/unpacking BeautifulSoup==3.2.0 ....
  Downloading BeautifulSoup-3.2.0.tar.gz
  Running setup.py egg_info for package BeautifulSoup
...
...
...

Storing complete log in /home/fabuser/.pip/pip.log 
```

## 使用 Fabric 重启 Apache 服务器

```py
We can use Fabric for many other things, we can also create a Fabric script for
restarting our **Apache servers.** 

That would look something like this 
```

```py
from fabric.api import *

env.hosts = ['[[email protected]](/cdn-cgi/l/email-protection):22']

def restart_apache2():
    sudo('service apache2 restart') 
```

```py
To restart apache2 on host 'host1.example.com', the only thing we have to do is 
```

```py
$fab restart_apache2 
```

## 使用结构发送电子邮件

```py
We aren't limited to hosts in "roledefs", we could also for example put in email
addresses. 
```

```py
from fabric.api import env, run

# Define sets of email addresses as role

env.roledefs = {
    'test': ['localhost'],
    'dev': ['[[email protected]](/cdn-cgi/l/email-protection)'],
    'staging': ['[[email protected]](/cdn-cgi/l/email-protection)'],
    'production': ['[[email protected]](/cdn-cgi/l/email-protection)']
} 

def deploy():
    run('echo test') 
```

```py
To run it a specific role use the -R switch 
```

```py
$ fab -R test deploy

[localhost] Executing task 'deploy' 
```

##### 进一步阅读

```py
[http://www.virtualenv.org/en/latest/](http://www.virtualenv.org/en/latest/ "virtualenv")
[http://www.pip-installer.org/en/latest/requirements.html](http://www.pip-installer.org/en/latest/requirements.html "requirements")
[http://docs.fabfile.org/en/1.6/](http://docs.fabfile.org/en/1.6/ "fabric")
[Stackoverflow](https://stackoverflow.com/questions/2326797/how-to-set-target-hosts-in-fabric-file "stackoverflow") 
```