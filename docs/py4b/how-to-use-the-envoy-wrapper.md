# 如何使用特使

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/how-to-use-the-envoy-wrapper>

## 关于特使

最近我偶然发现了特使。Envoy 是子流程模块的包装器，而
应该是 Python 的人性化子流程。

它是由 Kenneth Reitz(“[Requests:HTTP for Humans](http://docs.python-requests.org/en/latest/ "requests")”的作者)写的

## 为什么使用特使？

它被设计成一个易于使用的替代子流程的方法。

"特使:面向人类的 Python 子流程."

## 安装特使

特使可从 PyPI 获得，并可与 [pip](http://www.pip-installer.org/en/latest/ "pip") 一起安装。

通过 pip 命令行工具搜索 Envoy 包。

```py
pip search envoy
```

"特使-运行外部进程的简单 API . "安装特使

```py
$ pip install envoy
```

## 进口使节

就像任何其他 Python 模块一样，我们必须首先导入它们。

启动 Python 解释器并输入“import envoy”

```py
import envoy
```

太好了，特使是进口的，现在我们可以开始发现它的功能等。

## 特使方法和属性

在 Python 中导入一个模块后，看看该模块提供了哪些函数
、类和方法总是好的。一种方法是使用
“dir(envoy)”。

### 使用目录(模块)

这将列出在
Envoy 模块中定义的所有函数和变量的名称。

```py
>>> dir(envoy)
```

这将为您提供如下输出:

```py
['Command', 'ConnectedCommand', 'Response', '__builtins__', '__doc__', '__file__',
'__name__', '__package__', '__path__', '__version__', 'connect', 'core', 'expand_args',
'os', 'run', 'shlex', 'subprocess', 'threading’]
```

如果希望每行得到一个名字，只需运行一个简单的 for 循环:

```py
>>> for i in dir(envoy): print i
```

该输出每行显示一个名称:

```py
...
Command
ConnectedCommand
Response
__builtins__
__doc__
__file__
__name__
__package__
__path__
__version__
connect
core
expand_args
os
run
shlex
subprocess
threading
>>>
```

您还可以使用“help(envoy)”来获取所有功能的文档。

## 特使用法

让我们来看看特使的“运行”方法。

### envoy.run()

要检查我们机器的正常运行时间，我们可以使用“uptime”命令。

```py
r = envoy.run("uptime”)
```

### 标准输出

为了查看输出，我们添加了“std_out”:

```py
>>> r.std_out
'15:11  up 6 days,  1:04, 3 users, load averages: 0.55 0.57 0.61
‘
```

### 状态代码

要获取状态代码，请将“status_code”添加到对象中。

```py
print r.status_code
0
```

### 命令

运行命令，得到响应:

```py
>>> print r 
```

### 标准误差

要获得标准误差，请添加“std_err”。

```py
r.std_err
```

### 历史

获取历史。

```py
r.history
[<response 'uptime'="">]
```

### 特使示例

检查镀铬工艺

```py
r = envoy.run("ps aux |grep Chrome")
print r.std_out
```

在最后一个例子中，我使用了多个命令。

```py
import envoy

cmd = ['date', "uptime", "w"]

r = envoy.run(cmd)

print r.std_out
```

获取所有命令的状态代码

```py
import envoy

cmd = ['date', "uptime", "w"]

for command in cmd:
    r = envoy.run(cmd)
    print r.status_code
```

获取状态代码+每个命令的输出

```py
import envoy

cmd = ['date', "uptime", "w"]

for command in cmd:
    r = envoy.run(cmd)
    print r.status_code, r.std_out
```

特使已经成为我处理外部命令调用的主要库。

它被设计成一个易于使用的替代子流程的工具，envoy.run 的便利性真的很好。

##### 更多阅读

[https://github.com/kennethreitz/envoy](https://github.com/kennethreitz/envoy "envoy")
http://stackoverflow.com/search?q=envoy