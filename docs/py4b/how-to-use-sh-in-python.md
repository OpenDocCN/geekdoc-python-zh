# 如何在 Python 中使用 sh

> 原文：<https://www.pythonforbeginners.com/modules-in-python/how-to-use-sh-in-python>

## 什么是 sh？

sh 是一个独特的子进程包装器，它将你的系统程序动态映射到 Python
函数。 **sh** 通过给你
Bash 的良好特性(简单的命令调用，简单的管道)和 Python 的所有功能
和灵活性，帮助你用 Python 编写 shell 脚本。[ [来源](https://amoffat.github.io/sh/ "sh") ]

## 从 sh 开始

sh 是一个成熟的 Python 子进程接口，允许你调用任何 T2 程序，就像调用一个函数一样。与使用子进程相比，sh 可以让你调用任何你
可以从登录 shell 中运行的东西。更重要的是，可以让您更容易地捕获和解析输出。

## 装置

sh 的**安装**通过 **pip 命令**完成

```py
pip install sh 
```

## 使用

最简单的方法是直接导入 sh 或者从 sh 导入你的
程序。您想要运行的每个命令都像任何其他模块一样被导入。

该命令就像 Python 语句一样可用。

参数通常被传递，输出可以以类似于
的方式被捕获和处理。

```py
# get interface information
import sh
print sh.ifconfig("eth0")

from sh import ifconfig
print ifconfig("eth0")

# print the contents of this directory
print ls("-l")

# substitute the dash for an underscore for commands that have dashes in their names
sh.google_chrome("http://google.com”)
```

## 使用 sh 执行命令

命令就像函数一样被调用。

*"注意，这些不是 Python 函数，它们通过解析路径在系统上动态运行二进制命令
，就像 Bash 所做的那样。通过这种
方式，你系统上的所有程序都可以很容易地用 Python 实现。”*

许多程序都有自己的命令子集，比如 git (branch，checkout)。

sh 通过属性访问处理子命令。

```py
from sh import git

# resolves to "git branch -v"
print(git.branch("-v"))

print(git("branch", "-v")) # the same command
```

## 关键字参数

关键字参数也像您预期的那样工作:它们被替换为
长格式和短格式命令行选项。来源

```py
# Resolves to "curl http://duckduckgo.com/ -o page.html --silent"
sh.curl("http://duckduckgo.com/", o="page.html", silent=True)

# If you prefer not to use keyword arguments, this does the same thing
sh.curl("http://duckduckgo.com/", "-o", "page.html", "--silent")

# Resolves to "adduser amoffat --system --shell=/bin/bash --no-create-home"
sh.adduser("amoffat", system=True, shell="/bin/bash", no_create_home=True)

# or
sh.adduser("amoffat", "--system", "--shell", "/bin/bash", "--no-create-home”)
```

## 查找命令

“Which”查找程序的完整路径，如果不存在，则返回 None。

这个命令是作为 Python 函数实现的少数命令之一，因此
不依赖于实际存在的“哪个”程序。

```py
print sh.which("python")     # "/usr/bin/python"
print sh.which("ls")         # "/bin/ls"

if not sh.which("supervisorctl"): sh.apt_get("install", "supervisor", “-y”)
```

你可以在 sh 上使用更多的特性，你可以在[官方文档](https://amoffat.github.io/sh/#basic-features "basic_features")中找到所有的
。

## 烘烤

sh 能够将参数“烘焙”成命令。

```py
# The idea here is that now every call to ls will have the “-la” arguments already specified.
from sh import ls

ls = ls.bake("-la")
print(ls) # "/usr/bin/ls -la"

# resolves to "ls -la /"
print(ls(“/“))
```

##### 烘焙的 ssh 命令

在命令上调用“bake”会创建一个可调用的对象，该对象会自动传递
以及传递给“bake”的所有参数。

```py
# Without baking, calling uptime on a server would be a lot to type out:
serverX = ssh("myserver.com", "-p 1393", "whoami")

# To bake the common parameters into the ssh command
myserver = sh.ssh.bake("myserver.com", p=1393)

print(myserver) # "/usr/bin/ssh myserver.com -p 1393”
```

既然“myserver”可调用对象代表了一个烘焙的 ssh 命令，那么您可以轻松地调用服务器上的任何内容:

```py
# resolves to "/usr/bin/ssh myserver.com -p 1393 tail /var/log/dumb_daemon.log -n 100"
print(myserver.tail("/var/log/dumb_daemon.log", n=100))

# check the uptime
print myserver.uptime()
 15:09:03 up 61 days, 22:56,  0 users,  load average: 0.12, 0.13, 0.05
```

更多高级功能，请参见[官方文档](https://amoffat.github.io/sh/#advanced-features "advanced_features")。

##### 来源

[https://github.com/amoffat/sh](https://github.com/amoffat/sh "amoffat")
https://github.com/Byzantium/Byzantium