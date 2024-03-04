# 在 Python 中使用 Telnet

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-using-telnet>

## 在 Python 中使用 Telnet

```py
 To make use of Telnet in Python, we can use the telnetlib module. 

That module provides a Telnet class that implements the Telnet protocol.

The Telnet module have several methods, in this example I will make use of these:
read_until, read_all() and write() 
```

## Python 中的 Telnet 脚本

```py
 Let's make a telnet script 
```

```py
import getpass
import sys
import telnetlib

HOST = "hostname"

user = raw_input("Enter your remote account: ")

password = getpass.getpass()

tn = telnetlib.Telnet(HOST)

tn.read_until("login: ")

tn.write(user + "
")

if password:
    tn.read_until("Password: ")
    tn.write(password + "
")

tn.write("ls
")

tn.write("exit
")

print tn.read_all()

```

```py
 At ActiveState you can find more Python scripts using the telnetlib, 
for example [this](https://code.activestate.com/recipes/52228/ "code") script.

For more information about using the Telnet client in Python, please see the 
[official documentation](https://docs.python.org/2/library/telnetlib.html "python"). 
```