# 如何在 Python 中使用 Fabric

> 原文：<https://www.pythonforbeginners.com/systems-programming/how-to-use-fabric-in-python>

## 什么是面料？

```py
Fabric is a Python library and command-line tool for streamlining the use of SSH
for application deployment or systems administration tasks. 

Typical usage involves creating a Python module containing one or more functions,
then executing them via the fab command-line tool.

You can execute shell commands over SSH, so you only need to have SSH running on
the remote machine. It interact with the remote machines that you specify as if
they were local. 
```

## 装置

```py
Fabric requires Python version 2.5 or 2.6 (Fabric has not yet been tested on
Python 3.x)

The most common ways of installing Fabric is via, pip, easy_install or via
the operating system's package manager: 
```

```py
pip install fabric

sudo easy_install fabric

sudo apt-get install fabric	# (the package is typically called fabric or python-fabric.) 
```

```py
Please read the [official](http://docs.fabfile.org/en/1.5/installation.html "fabfile") documentation for more information (dependancies etc..) 
```

## 织物使用

```py
On their [website](http://docs.fabfile.org/en/1.5/ "fabfile_doc") they write:

"it provides a basic suite of operations for executing local or remote shell
commands (normally or via sudo) and uploading/downloading files, as well as
auxiliary functionality such as prompting the running user for input, or
aborting execution"

Having that information, let's move on. 
```

```py
The installation process added a Python script called fab to a directory in
your path. 

This is the script which will be used to make everything happen with Fabric. 

Just running fab from the command-line won't do much at all.

In order to do anything interesting, we'll need to create our first fabfile. 

Before we do that I would like to show some of the functions in Fabric. 
```

## 织物功能

```py
Fabric provides a set of commands in fabric.api that are simple but powerful.

With Fabric, you can use simple Fabric calls like 
```

```py
local	# execute a local command)
run	# execute a remote command on all specific hosts, user-level permissions)
sudo	# sudo a command on the remote server)
put	# copy over a local file to a remote destination)
get	# download a file from the remote server)
prompt	# prompt user with text and return the input (like raw_input))
reboot	# reboot the remote system, disconnect, and wait for wait seconds) 
```

## 证明

```py
Fabric relies on the SSH Model, you can use SSH Keys but you can also control
access to root via sudoers. 

The user on the server does not need to be added to "~/.ssh/authorized_keys",
but if it is you don't have to type the password every time you want to execute
a command.

If you ever have to disable the access to a user, just turn off their SSH account. 
```

## Fabfile

```py
You can load Python modules with Fabric. 

By default, it looks for something named either fabfile or fabfile.py. 

This is the file that Fabric uses to execute tasks. 

Each task is a simple function.

The fabfile should be in the same directory where you run the Fabric tool. 

The fabfile is where all of your functions, roles, configurations, etc. will
be defined. 

It's just a little bit of Python which tells Fabric exactly what it needs to do. 

The "fabfile.py" file only has to be stored on your client. 

An SSH server like OpenSSH needs to be installed on your server and an
SSH client needs to be installed on your client. 
```

## 创建 Fabfile

```py
To start just create a blank file called fabfile.py in the directory you’d like
to use the fabric commands from.

You basically write rules that do something and then you (can) specify on which
servers the rules will run on. 

Fabric then logs into one or more servers in turn and executes the shell commands 
defined in "fabfile.py". 

If you are located in the same dir as "fabfile.py" you can go **"fab --list"**
to see a list of available commands and then "fab [COMMAND_NAME]" to execute a
command.

From [https://github.com/fabric/fabric](https://github.com/fabric/fabric "fabric_git")

Below is a small but complete "fabfile" containing a single task: 
```

```py
from fabric.api import run
def host_type():
    run('uname -s') 
```

```py
Once a task is defined, it may be run on one or more servers, like so 
```

```py
**$ fab -H localhost,linuxbox host_type**

[localhost] run: uname -s
[localhost] out: Darwin
[linuxbox] run: uname -s
[linuxbox] out: Linux

Done.
Disconnecting from localhost... done.
Disconnecting from linuxbox... done. 
```

```py
You can run fab -h for a full list of command line options

In addition to use via the fab tool, Fabric's components may be imported into
other Python code, providing a Pythonic interface to the SSH protocol suite at
a higher level than that provided by e.g. the ssh library, 
(which Fabric itself uses.) 
```

## 连接到远程服务器

```py
As you can see above, Fabric will look for the function host_type in the
fabfile.py of the current working directory. 
```

```py
In the next example we can see how the Fabric Api uses an internal env variable
to manage information for connecting to remote servers.

The user is the user name used to login remotely to the servers and the hosts is
a list of hosts to connect to. 

Fabric will use these values to connect remotely for use with the run
and sudo commands. 

It will prompt you for any passwords needed to execute commands or connect to
machines as this user. 
```

```py
# First we import the Fabric api
from fabric.api import *

# We can then specify host(s) and run the same commands across those systems
env.user = 'username'

env.hosts = ['serverX']

def uptime():
    run("uptime") 
```

```py
This will first load the file ~/fabfile.py

Compiling the file into ~/fabfile.pyc # a standard pythonism

Connect via SSH to the host 'serverX'

Using the username 'username'

Show the output of running the command "uptime" upon that remote host.

Now, when we run the fabfile, we can see that we can connect to the remote server.
$ fab uptime 
```

```py
If you have different usernames on different hosts, you can use: 
```

```py
env.user = 'username'
env.hosts = ['[[email protected]](/cdn-cgi/l/email-protection)', 'serverX'] 
```

```py
Now userX username would be used on 192.168.1.1 and 'username' would be used
on 'serverX' 
```

## 角色

```py
You can define roles in Fabric, and only run actions on servers tied to a
specific role. 

This script will run get_version on hosts members of the role "webservers",
by running first on www1, then www2 etc. 
```

```py
fab -R webservers 
```

```py
from fabric.api import *

# Define sets of servers as roles

env.roledefs = {
    'webservers': ['www1', 'www2', 'www3', 'www4', 'www5'],
    'databases': ['db1', 'db2']
}

# Set the user to use for ssh
env.user = 'fabuser'

# Restrict the function to the 'webservers' role

@roles('webservers')

def get_version():
    run('cat /etc/issue') 
```

```py
# To run get_version on both roles (webservers and databases); 
```

```py
$ fab get_version 
```

```py
@roles ('webservers', 'databases')

    def get_version():

 run('cat /etc/issue') 
```

```py
Any function you write in your fab script can be assigned to one or more roles. 

You can also include a single server in multiple roles. 
```

## 将目录复制到远程机器

```py
I would like to end this introduction of Fabric, by showing an example of
how to copy a directory to a remote machine. 
```

```py
from fabric.api import *

env.hosts = ['[[email protected]](/cdn-cgi/l/email-protection)']

def copy():
    # make sure the directory is there!
    run('mkdir -p /home/userX/mynewfolder')

    # our local 'localdirectory' (it may contain files or subdirectories)
    put('localdirectory', '/home/userX/mynewfolder') 
```

## 结论

```py
Fabric can be used for many things, including deploying, restarting servers,
stopping and restarting processes. 

You write the description of what is to be done in Python and Fabric takes
care of executing it on all the machines you specify. 

Once you've used it a bunch of times you'll accumulate many "fab files" that
you can re-use.

Basically, any time we would need to log in to multiple servers to do something,
you can use Fabric. 

It’s simple and powerful and the documentation is really good. 
```

##### 进一步阅读

```py
[http://docs.fabfile.org/en/1.5/index.html](http://docs.fabfile.org/en/1.5/index.html "fabfile_official")
[https://gist.github.com/DavidWittman/1886632](https://gist.github.com/DavidWittman/1886632 "https://gist.github.com/DavidWittman/1886632")
[http://mattsnider.com/using-fabric-for-painless-scripting/](http://mattsnider.com/using-fabric-for-painless-scripting/ "http://mattsnider.com/using-fabric-for-painless-scripting/")
[http://www.clemesha.org/blog/modern-python-hacker-tools-virtualenv-fabric-pip/](http://www.clemesha.org/blog/modern-python-hacker-tools-virtualenv-fabric-pip/ "http://www.clemesha.org/blog/modern-python-hacker-tools-virtualenv-fabric-pip/")
[http://code.mixpanel.com/2010/09/09/easy-python-deployment-with-fabric-and-git/](https://code.mixpanel.com/2010/09/09/easy-python-deployment-with-fabric-and-git/ "http://code.mixpanel.com/2010/09/09/easy-python-deployment-with-fabric-and-git/")
[http://stackoverflow.com/questions/6308686/understanding-fabric/9894988#9894988](https://stackoverflow.com/questions/6308686/understanding-fabric/9894988#9894988 "http://stackoverflow.com/questions/6308686/understanding-fabric/9894988#9894988") 
```